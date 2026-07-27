"""Web layer: gradio.Server app + WebSocket real-time A/V session loop.

``build_app(engine_factory, brain_factory, face_cropper_factory)`` returns a
``gradio.Server`` instance (a FastAPI subclass) with:

    GET  /            -> static/index.html
    GET  /defaults    -> JSON {system_prompt, max_prompt_chars} for the UI editor
    GET  /healthz     -> JSON status (lease held? session busy?)
    API  /run_session -> holds the ZeroGPU lease, streams ready/tick/expired
    WS   /ws          -> the binary real-time protocol from DESIGN.md

This module is deliberately framework-clean: it never imports the mocks nor the
real engine/brain. Whatever the factories return is used, so swapping
``server.engine.AvatarEngine`` / ``server.conversation.ConversationBrain`` in
requires no change here.

WebSocket protocol (little-endian, first byte = tag)
---------------------------------------------------
  C->S 0x01  mic audio     : [tag u8][int16 PCM mono 16 kHz ...]   (~100 ms)
  C->S 0x02  webcam frame  : [tag u8][JPEG bytes]                  (~25 fps)
  C->S 0x03  reference img : [tag u8][encoded image bytes]         (once, see below)
  S->C 0x11  avatar audio  : [tag u8][block_idx u32][int16 PCM x 6400]
  S->C 0x12  avatar frame  : [tag u8][block_idx u32][frame_in_block u8][JPEG]
  both  JSON text          : control + ConversationBrain events, plus
                             {"type":"ready"} once priming is done and
                             {"type":"error","code":"busy"} for extra sessions.

Session handshake (per-session config, before priming)
------------------------------------------------------
Right after ``{"type":"hello"}`` the client sends exactly one::

    C->S {"type":"session_config",
          "voice_id": "<ElevenLabs id>" | "",   # "" -> the server's VOICE_ID env
          "system_prompt": "<LLM system prompt>" | "",   # "" -> the built-in one
          "reference": true|false}              # true -> a 0x03 frame follows

and, when ``reference`` is true, immediately follows it with one 0x03 binary
frame carrying the raw upload (<= ``MAX_REF_BYTES``). Ordering is guaranteed by
the websocket, so the reference always precedes the first 0x02 webcam frame.

``system_prompt`` is trimmed and truncated to ``MAX_PROMPT_CHARS``. The server's
built-in default travels the other way, in the opening ``hello``, so the client
can seed its editor with the real prompt rather than a blank box::

    S->C {"type":"hello","server":...,"fps":...,"system_prompt":"<default>"}

Everything downstream is gated on that handshake: the brain is built *after* it
(so ``voice_id`` / ``system_prompt`` reach ``brain_factory``) and
``engine.set_reference`` runs on the GPU thread *before* the session's
``start_session``, because the identity latents are engine-level state shared by
every session. The server answers with ``{"type":"reference","ok":true|false}``
and restores the default reference when the session ends. A client that sends no
``session_config`` is not broken: after ``CONFIG_TIMEOUT`` the session proceeds
on the server defaults.

``voice_id`` and ``system_prompt`` are held in memory for the session only —
never logged, never echoed back, never written to ``/healthz`` (which reports
bools, not the values).

Concurrency model (one WS session at a time)
-------------------------------------------
  * ``_GPU_EXECUTOR``  – ONE thread, module-global. Every call that reaches the
    GPU is submitted there, so the GPU is never re-entered concurrently even
    across reconnects.
  * ``_CROP_EXECUTOR`` – ONE thread, submitting webcam frames for cropping (the
    real cropper keeps EMA state, so it must stay single-threaded).
  * a single writer task owns the WebSocket; every other task enqueues frames,
    so we never interleave two concurrent ``send_bytes`` calls.

ZeroGPU
-------
There is no GPU in this process. Both executors above now hand their work to
``server.gpu_session.PROXY``, which forwards it over fork-context queues to a
worker holding a ``@spaces.GPU`` lease — see ``server/gpu_session.py`` for why
the state cannot live here. Three consequences show up in this file:

  * the frame ring, the face cropper and JPEG **encoding** moved into the
    worker, so ``step`` returns ready-made JPEGs and ``_crop_task`` only ships
    the ~25 KB webcam frame across;
  * a session cannot start until the client has acquired a lease (the
    ``/run_session`` Gradio endpoint, which is also what attaches the visitor's
    ``X-IP-Token`` so their own quota is billed rather than the Space's IP); and
  * the lease expires on a wall clock, so any GPU call can raise ``LeaseError``
    and that ends the session cleanly rather than crashing it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from server import gpu_session
from server.gpu_session import LeaseError

log = logging.getLogger("avatar.web")

# ---------------------------------------------------------------- protocol --- #
TAG_MIC = 0x01
TAG_CAM = 0x02
TAG_REF = 0x03
TAG_AUDIO_BLOCK = 0x11
TAG_VIDEO_FRAME = 0x12

# ------------------------------------------------------------------- tuning -- #
FPS = 25
SR = 16000
BLOCK_FRAMES = 10
BLOCK_SAMPLES = 6400            # 400 ms @ 16 kHz
BLOCK_SEC = BLOCK_FRAMES / FPS  # 0.4
MIC_RING_SEC = 8.0
# generate at most ~N blocks ahead of real time (env: AVATAR_LEAD_BLOCKS)
LEAD_BLOCKS = float(os.environ.get("AVATAR_LEAD_BLOCKS", "1.0"))
SNAPSHOT_PERIOD = float(os.environ.get("AVATAR_SNAPSHOT_PERIOD", "1.0"))
PACING_LOG_BLOCKS = int(os.environ.get("AVATAR_PACING_LOG_BLOCKS", "25"))  # ~10 s
# Fall this far behind real time and the block loop rebases instead of trying to
# catch up -- see the resync branch in _Session.generator().
RESYNC_SEC = float(os.environ.get("AVATAR_RESYNC_SEC", "1.2"))
# Re-prime the engine (fresh KV cache) this often, but only during a silent block
# while the avatar is not speaking. Superseded by the engine's pose anchor
# (engine.AvatarEngine._anchor_pose, AVATAR_POSE_GAIN) and therefore off by
# default; kept as a fallback for reference identities where the anchor's
# |r_s|-relative deadband might not transfer.
#
# Rollout drift measured on 240 s of the repo's real inputs, sharpness as % of
# the first 20 s, in 20 s windows:
#   no mitigation  89 76 60 52 50 45 41 45 40 42 39 45
#   re-prime 40 s  89 76 69 63 88 75 73 55 82 73 72 54   <- the resets are visible
#   pose anchor    92 94 85 86 83 80 88 83 76 71 78 77
# The re-prime row is a sawtooth by construction: it restarts the rollout from
# the reference state, so the avatar snaps back to a reference-like pose every
# 40 s and the 400 ms morph reads as a cut. It also cannot fire at all while the
# avatar speaks continuously, which is exactly when drift is unchecked. 0 disables.
REPRIME_SECS = float(os.environ.get("AVATAR_REPRIME_SECS", "0"))
# AVATAR_JPEG_QUALITY / AVATAR_JPEG_SUBSAMPLING / AVATAR_REPRIME_XFADE /
# AVATAR_CAM_WARM_SIZE are read in server/gpu_session.py now -- encoding, the
# re-prime dissolve and the cropper warm-up all happen in the GPU worker.
SEND_QUEUE_MAX = 240            # ~9 blocks of video+audio; video dropped if full
# How long the session waits for the client's {"type":"session_config"} before
# falling back to the server defaults. Our own client sends it in the same tick
# as `hello`, so this only ever pays out for third-party / test clients.
CONFIG_TIMEOUT = float(os.environ.get("AVATAR_CONFIG_TIMEOUT", "3.0"))
# Upper bound on an uploaded reference image. index.html already re-encodes to
# <=1024 px JPEG q0.9 (~200 KB); this is the guard against everything else.
MAX_REF_BYTES = int(os.environ.get("AVATAR_MAX_REF_BYTES", str(8 * 1024 * 1024)))
# Cap on a user-supplied system prompt. The prompt is prepended to every LLM
# request, so an unbounded one would cost time-to-first-token on every turn;
# 8 KB is ~25x the built-in prompt and still far under the model's context.
MAX_PROMPT_CHARS = int(os.environ.get("AVATAR_MAX_PROMPT_CHARS", "8192"))

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

# Module-global executors: shared across connections so a reconnect can never
# create a second GPU thread. Both now submit RPCs to the leased worker rather
# than touching CUDA; the JPEG pool moved into the worker with the encoding.
_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
_CROP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crop")

# Conversation carried across a lease boundary, for the client's "Keep going".
# A ZeroGPU lease is reclaimed every SESSION_SECONDS, so without this every
# session starts amnesiac. Only the LLM turn list travels; the brain, its
# ElevenLabs socket and the video rollout are all rebuilt. Kept module-level
# (not per-connection) precisely because the WebSocket dies with the lease.
#
# One slot is enough: the Space serves one session at a time. It holds user
# speech and avatar replies, so it expires rather than lingering indefinitely.
CARRYOVER_TTL = float(os.environ.get("AVATAR_CARRYOVER_TTL", "900"))  # 15 min
_CARRYOVER: dict[str, Any] = {"history": [], "at": 0.0}


# =========================================================================== #
# small helpers
# =========================================================================== #
class MicRing:
    """Fixed-size float32 ring buffer for the incoming mic stream."""

    def __init__(self, seconds: float = MIC_RING_SEC, sr: int = SR) -> None:
        self.n = int(seconds * sr)
        self._buf = np.zeros(self.n, dtype=np.float32)
        self._written = 0

    def push(self, pcm_f32: np.ndarray) -> None:
        x = np.asarray(pcm_f32, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        if x.size >= self.n:
            self._buf[:] = x[-self.n:]
            self._written += x.size
            return
        pos = self._written % self.n
        end = pos + x.size
        if end <= self.n:
            self._buf[pos:end] = x
        else:
            k = self.n - pos
            self._buf[pos:] = x[:k]
            self._buf[: end - self.n] = x[k:]
        self._written += x.size

    def last(self, n: int) -> np.ndarray:
        """Most recent ``n`` samples, zero-padded at the front if short."""
        out = np.zeros(n, dtype=np.float32)
        have = min(n, self._written, self.n)
        if have:
            pos = self._written % self.n
            start = pos - have
            if start >= 0:
                out[n - have:] = self._buf[start:pos]
            else:
                out[n - have: n - have - start] = self._buf[start:]
                out[n - have - start:] = self._buf[:pos]
        return out


# NOTE: _jpeg_bytes_to_rgb / _rgb_to_jpeg / _xfade used to live here. They moved
# into server/gpu_session.py's worker: decoding and cropping happen next to the
# GPU that needs the result, and encoding happens next to the frames that would
# otherwise have to be pickled across the fork as 16 MB of raw uint8 per block.


def _pack_audio(block_idx: int, pcm_i16: np.ndarray) -> bytes:
    a = np.asarray(pcm_i16)
    if a.dtype != np.int16:
        a = np.clip(a, -32768, 32767).astype(np.int16)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return struct.pack("<BI", TAG_AUDIO_BLOCK, block_idx) + a.tobytes()


def _pack_video(block_idx: int, frame_in_block: int, jpeg: bytes) -> bytes:
    return struct.pack("<BIB", TAG_VIDEO_FRAME, block_idx, frame_in_block) + jpeg


# =========================================================================== #
# lease holder
# =========================================================================== #
class _LeaseHolder:
    """Tracks whether a ZeroGPU lease is currently held.

    On the dedicated Space this class built and ``load()``ed the engine once, in
    the GPU thread. On ZeroGPU the engine is built at import (``app.py``, where
    ZeroGPU packs its weights) and warmed inside the lease, so nothing here
    touches the model: it only reports whether ``/run_session`` has a worker up,
    and hands out the proxy that talks to it.
    """

    def __init__(self) -> None:
        self.load_error: str | None = None
        self.warm_seconds: float | None = None
        self.session_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return gpu_session.PROXY.live

    def on_ready(self, warm_seconds: float, session_seconds: float) -> None:
        self.warm_seconds = warm_seconds
        self.session_seconds = session_seconds
        self.load_error = None

    async def get(self) -> Any:
        """Return the live proxy, or explain why there isn't one."""
        proxy = gpu_session.PROXY
        if not proxy.live:
            raise LeaseError(proxy.last_error
                             or "no GPU lease — call /run_session first")
        return proxy


# =========================================================================== #
# one WebSocket session
# =========================================================================== #
class _Session:
    def __init__(self, ws: WebSocket, gpu: Any,
                 brain_factory: Callable[..., Any]) -> None:
        self.ws = ws
        # The leased worker. Owns the engine, the cropper and the frame ring;
        # every attribute this class used to read off `engine` now arrives as an
        # RPC result instead.
        self.gpu = gpu
        # Built by configurator(), not here: the client's voice_id has to reach
        # brain_factory, and it only arrives with `session_config`.
        self.brain_factory = brain_factory
        self.brain: Any = None
        self.loop = asyncio.get_running_loop()

        self.mic = MicRing()
        # The cropped-frame ring lives in the worker now; all this side needs to
        # know is whether at least one frame has landed, so the primer can go.
        self.latest_jpeg: bytes | None = None  # newest raw webcam JPEG
        self.have_first_frame = False
        self.primed = asyncio.Event()
        self.closing = asyncio.Event()
        self.lease_lost: str | None = None

        # per-session config handshake (see the protocol docstring)
        self.session_cfg: dict[str, Any] = {}  # holds voice_id; never logged
        self.resume = False                    # continue the previous conversation
        self.resumed_messages = 0
        self.ref_bytes: bytes | None = None
        self.config_ready = asyncio.Event()    # session_config (+ 0x03) received
        self.configured = asyncio.Event()      # config applied; brain/primer may go
        self.ref_applied = False

        self.send_q: asyncio.Queue[tuple[int, Any]] = asyncio.Queue(maxsize=SEND_QUEUE_MAX)
        self._crop_busy = False
        self._snapshot_at = 0.0

        # stats
        self.t_start = time.monotonic()
        self.block_idx = 0
        self.step_ms_max = 0.0
        self._log_at = 0
        self._log_step_ms = 0.0
        self._log_gpu_ms = 0.0
        self.gpu_step_ms = 0.0
        self.n_mic_chunks = 0
        self.n_cam_frames = 0
        self.n_cropped = 0
        self.dropped_video = 0
        self.step_ms_total = 0.0
        self.late_blocks = 0
        self.n_reprimes = 0
        self.n_resyncs = 0
        self._last_prime_at = 0.0
        # Last engine-side telemetry seen in a step reply, so /healthz can report
        # pose drift without paying an RPC per request.
        self._engine_stats: dict = {}

    # ---------------- outbound ---------------- #
    def enqueue(self, payload: bytes | dict, *, droppable: bool = False) -> None:
        kind = 1 if isinstance(payload, (bytes, bytearray)) else 0
        try:
            self.send_q.put_nowait((kind, payload))
        except asyncio.QueueFull:
            if droppable:
                self.dropped_video += 1
            else:  # never drop audio or events: make room by discarding oldest
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.send_q.get_nowait()
                    self.dropped_video += 1
                with contextlib.suppress(asyncio.QueueFull):
                    self.send_q.put_nowait((kind, payload))

    def on_brain_event(self, ev: dict) -> None:
        """Called from any thread by the ConversationBrain."""
        try:
            self.loop.call_soon_threadsafe(self.enqueue, ev)
        except RuntimeError:
            pass  # loop already closed

    async def writer(self) -> None:
        while True:
            kind, payload = await self.send_q.get()
            if kind == 1:
                await self.ws.send_bytes(payload)
            else:
                await self.ws.send_text(json.dumps(payload))

    # ---------------- inbound ---------------- #
    async def reader(self) -> None:
        while True:
            msg = await self.ws.receive()
            typ = msg.get("type")
            if typ == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))
            data = msg.get("bytes")
            if data is not None:
                self._on_binary(data)
                continue
            text = msg.get("text")
            if text:
                self._on_text(text)

    def _on_binary(self, data: bytes) -> None:
        if not data:
            return
        tag = data[0]
        if tag == TAG_MIC:
            body = data[1:]
            if len(body) < 2:
                return
            if len(body) % 2:
                body = body[: len(body) - 1]
            pcm16 = np.frombuffer(body, dtype="<i2")
            self.mic.push(pcm16.astype(np.float32) / 32768.0)
            self.n_mic_chunks += 1
            if self.brain is None:      # mic can arrive before the handshake lands
                return
            try:
                self.brain.feed_user_audio(pcm16)
            except Exception:
                log.exception("brain.feed_user_audio failed")
        elif tag == TAG_REF:
            body = bytes(data[1:])
            if len(body) > MAX_REF_BYTES:
                log.warning("reference image too large (%d bytes), ignored", len(body))
                self.enqueue({"type": "reference", "ok": False,
                              "text": f"Image too large (max {MAX_REF_BYTES // (1024*1024)} MB)."})
            else:
                self.ref_bytes = body or None
            self.config_ready.set()
        elif tag == TAG_CAM:
            jpeg = bytes(data[1:])
            if not jpeg:
                return
            self.latest_jpeg = jpeg
            self.n_cam_frames += 1
            self._maybe_snapshot(jpeg)
            if not self._crop_busy:
                self._crop_busy = True
                asyncio.create_task(self._crop_task(jpeg))
        else:
            log.debug("unknown client tag 0x%02x", tag)

    def _on_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except Exception:
            return
        t = msg.get("type")
        if t == "ping":
            self.enqueue({"type": "pong", "t": msg.get("t")})
        elif t == "hello":
            log.info("client hello: %s", msg)
        elif t == "session_config":
            self.resume = bool(msg.get("resume"))
            voice_id = str(msg.get("voice_id") or "").strip()
            if voice_id:
                self.session_cfg["voice_id"] = voice_id     # NOT logged
            prompt = str(msg.get("system_prompt") or "").strip()[:MAX_PROMPT_CHARS]
            if prompt:
                self.session_cfg["system_prompt"] = prompt  # NOT logged
            # `reference: true` promises a 0x03 frame next; wait for it instead.
            if not msg.get("reference"):
                self.config_ready.set()

    def _maybe_snapshot(self, jpeg: bytes) -> None:
        now = time.monotonic()
        if now - self._snapshot_at >= SNAPSHOT_PERIOD:
            self._snapshot_at = now
            if self.brain is None:
                return
            try:
                self.brain.set_user_snapshot(jpeg)
            except Exception:
                log.exception("brain.set_user_snapshot failed")

    async def _crop_task(self, jpeg: bytes) -> None:
        """Ship the newest webcam JPEG to the worker to decode + crop.

        Only the ~25 KB JPEG crosses the fork; the 768 KB cropped result stays in
        the worker's ring, where ``step`` will read it.
        """
        try:
            res = await self.loop.run_in_executor(
                _CROP_EXECUTOR, self.gpu.push_frame, jpeg)
            self.n_cropped = res.get("n_cropped", self.n_cropped + 1)
            self.have_first_frame = True
        except LeaseError as exc:
            self._on_lease_lost(exc)
        except Exception:
            log.exception("crop failed")
        finally:
            self._crop_busy = False

    def _on_lease_lost(self, exc: Exception) -> None:
        """The GPU went away (expiry, crash, or the visitor's quota ran out)."""
        if self.lease_lost is None:
            self.lease_lost = str(exc)
            log.info("GPU lease lost: %s", exc)
            self.enqueue({"type": "lease_ended", "text": str(exc)})
        self.closing.set()

    async def closer(self) -> None:
        """Turn "the lease died" into an exception the task group reacts to.

        The session's tasks are awaited with FIRST_EXCEPTION, which a task
        *returning* does not satisfy. Without this, a generator() that bailed on
        LeaseError left the rest of the session running: the crop task kept
        shipping frames at 25 fps, each raising against a dead worker, until the
        client happened to disconnect.
        """
        await self.closing.wait()
        raise LeaseError(self.lease_lost or "session closing")

    # ---------------- generation ---------------- #
    def _want_reprime(self, avatar_pcm16: np.ndarray) -> bool:
        """True when it is safe *and* time to reset the generation drift.

        Only during a fully silent avatar-audio block and outside the 'speaking'
        state, so the pose cut never lands in the middle of a spoken word.
        """
        if REPRIME_SECS <= 0 or self.block_idx <= 1:
            return False
        if time.monotonic() - self._last_prime_at < REPRIME_SECS:
            return False
        if int(np.abs(avatar_pcm16).max(initial=0)) > 200:      # ~ -44 dBFS
            return False
        return getattr(self.brain, "state", "listening") != "speaking"

    async def generator(self) -> None:
        """Paced 400 ms loop: pull audio, step the engine, ship the block."""
        await self.primed.wait()
        t0 = time.monotonic()
        while True:
            # pace: do not run further than LEAD_BLOCKS ahead of the wall clock
            deadline = t0 + (self.block_idx - LEAD_BLOCKS) * BLOCK_SEC
            delay = deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            elif -delay > RESYNC_SEC:
                # Far enough behind that catching up is hopeless: at ~230 ms of
                # work per 400 ms block we claw back 170 ms each, so a single
                # multi-second stall keeps the stream lagging for a minute+.
                # The client indexes video by block, so it discards everything
                # that arrives that far behind its audio clock -- the avatar
                # freezes on the reference photo while the voice plays on.
                # Re-baseline instead: drop the debt, resume at real-time
                # cadence. Block indices are untouched, so the client stays in
                # sync with itself.
                log.info("pacing resync: %.1fs behind at block %d, rebasing",
                         -delay, self.block_idx)
                t0 = time.monotonic() - (self.block_idx - LEAD_BLOCKS) * BLOCK_SEC
                self.n_resyncs += 1

            b = self.block_idx
            self.block_idx += 1

            try:
                avatar_pcm16 = np.asarray(
                    self.brain.pull_avatar_audio(BLOCK_SAMPLES) if self.brain is not None
                    else np.zeros(BLOCK_SAMPLES, dtype=np.int16))
            except Exception:
                log.exception("brain.pull_avatar_audio failed")
                avatar_pcm16 = np.zeros(BLOCK_SAMPLES, dtype=np.int16)
            if avatar_pcm16.dtype != np.int16:
                avatar_pcm16 = np.clip(avatar_pcm16, -32768, 32767).astype(np.int16)
            if avatar_pcm16.size != BLOCK_SAMPLES:
                fixed = np.zeros(BLOCK_SAMPLES, dtype=np.int16)
                fixed[: min(BLOCK_SAMPLES, avatar_pcm16.size)] = \
                    avatar_pcm16[:BLOCK_SAMPLES]
                avatar_pcm16 = fixed

            # audio goes out first so the client can keep its clock running even
            # if video encoding lags
            self.enqueue(_pack_audio(b, avatar_pcm16))

            avatar_f32 = avatar_pcm16.astype(np.float32) / 32768.0
            user_f32 = self.mic.last(BLOCK_SAMPLES)

            tic = time.perf_counter()
            try:
                if self._want_reprime(avatar_pcm16):
                    # Drift reset: a fresh start_session() returns the 50 priming
                    # frames; the last 10 are this block's output, so the 400 ms
                    # cadence and the block indices are unaffected. The worker
                    # owns the cross-dissolve that hides the pose cut.
                    res = await self.loop.run_in_executor(
                        _GPU_EXECUTOR, self.gpu.reprime)
                    self._last_prime_at = time.monotonic()
                    self.n_reprimes += 1
                    log.info("re-primed engine at block %d (drift reset #%d)",
                             b, self.n_reprimes)
                else:
                    res = await self.loop.run_in_executor(
                        _GPU_EXECUTOR, self.gpu.step, avatar_f32, user_f32)
            except LeaseError as exc:
                self._on_lease_lost(exc)
                return
            jpegs = res.get("jpegs", [])
            # Round-trip time, not the engine's own step: the difference is the
            # queue hop + pickle, which is what we actually have to fit in 400 ms.
            step_ms = (time.perf_counter() - tic) * 1000.0
            self.gpu_step_ms = float(res.get("step_ms", 0.0))
            self.step_ms_total += step_ms
            self._log_step_ms += step_ms
            self._log_gpu_ms += self.gpu_step_ms
            self.step_ms_max = max(self.step_ms_max, step_ms)

            for i, jpg in enumerate(jpegs):
                self.enqueue(_pack_video(b, i, jpg), droppable=True)

            if time.monotonic() - (t0 + (b + 1) * BLOCK_SEC) > 0.2:
                self.late_blocks += 1

            # periodic pacing telemetry (integration/debug: see reports/integration.md)
            n = self.block_idx - self._log_at
            if PACING_LOG_BLOCKS and n >= PACING_LOG_BLOCKS:
                # gpu_ms is the engine's own step; step_ms adds the fork
                # round-trip. Both matter: the budget is 400 ms end to end, and
                # the gap between them is what the ZeroGPU port costs.
                self._engine_stats = self.gpu.stats()
                log.info(
                    "pacing: blocks=%d step_ms avg=%.0f gpu_ms avg=%.0f (window) "
                    "max=%.0f late=%d dropped_video=%d sendq=%d reprimes=%d "
                    "pose_d=%.1f/%.1f pose_corr=%d brain=%s",
                    self.block_idx, self._log_step_ms / n, self._log_gpu_ms / n,
                    self.step_ms_max,
                    self.late_blocks, self.dropped_video, self.send_q.qsize(),
                    self.n_reprimes, self._engine_stats.get("pose_dist", 0.0),
                    self._engine_stats.get("pose_deadband", 0.0),
                    self._engine_stats.get("pose_corrections", 0),
                    getattr(self.brain, "state", "?"),
                )
                self._log_at = self.block_idx
                self._log_step_ms = 0.0
                self._log_gpu_ms = 0.0
                self.step_ms_max = 0.0

    async def configurator(self) -> None:
        """Apply the client's per-session config, then release brain + primer.

        The reference swap is engine-level state, so the ordering here is
        load-bearing: ``set_reference`` is queued on ``_GPU_EXECUTOR`` and
        awaited *before* ``configured`` is set, hence strictly before this
        session's ``start_session()`` and its first ``step()``. Off the 400 ms
        path entirely — it happens once, while the client is still opening its
        camera.
        """
        try:
            await asyncio.wait_for(self.config_ready.wait(), CONFIG_TIMEOUT)
        except asyncio.TimeoutError:
            log.info("no session_config in %.1fs — using server defaults", CONFIG_TIMEOUT)

        try:
            if self.ref_bytes is not None:
                secs = await self.loop.run_in_executor(
                    _GPU_EXECUTOR, self.gpu.set_reference, self.ref_bytes)
                self.ref_applied = True
                log.info("custom reference applied (%.0f KB) in %.0f ms",
                         len(self.ref_bytes) / 1024.0, secs * 1000.0)
                self.enqueue({"type": "reference", "ok": True})
            elif not self.gpu.reference_is_default:
                # Previous session uploaded one and died before restoring it.
                # (A fresh lease always starts on the default, so this only
                # fires when two sessions share one lease.)
                secs = await self.loop.run_in_executor(
                    _GPU_EXECUTOR, self.gpu.set_reference, None)
                log.info("restored default reference in %.0f ms", secs * 1000.0)
        except LeaseError as exc:
            self._on_lease_lost(exc)
        except Exception as exc:
            log.exception("set_reference failed")
            self.enqueue({"type": "reference", "ok": False,
                          "text": f"Could not use that photo ({type(exc).__name__}); "
                                  f"falling back to the default face."})
        finally:
            self.configured.set()

    async def brain_starter(self) -> None:
        """Build + bring up the ConversationBrain, concurrently with priming.

        ``brain.start()`` costs a lazy import of the ElevenLabs/OpenAI SDKs plus
        a thread hand-off. Awaiting it before the reader/primer tasks exist would
        stall webcam ingestion (and therefore ``ready``) for its whole duration,
        so it gets its own task. Everything the brain exposes is safe to call
        before ``start()``: ``feed_user_audio`` only buffers, ``set_user_snapshot``
        only stores, ``pull_avatar_audio`` returns silence.
        """
        await self.configured.wait()        # session_cfg carries the voice id
        t0 = time.perf_counter()
        try:
            self.brain = self.brain_factory(self.on_brain_event, self.session_cfg)
            # Seed BEFORE start(): the first user turn must already see the
            # earlier conversation, or the avatar reintroduces itself.
            if self.resume and hasattr(self.brain, "import_history"):
                fresh = time.monotonic() - _CARRYOVER["at"] < CARRYOVER_TTL
                if fresh and _CARRYOVER["history"]:
                    self.resumed_messages = self.brain.import_history(
                        _CARRYOVER["history"])
                    log.info("resumed conversation: %d messages carried over",
                             self.resumed_messages)
                    self.enqueue({"type": "resumed",
                                  "messages": self.resumed_messages})
                else:
                    log.info("resume requested but no carryover within %.0fs",
                             CARRYOVER_TTL)
                    self.enqueue({"type": "resumed", "messages": 0})
            elif not self.resume:
                # Explicitly starting over: drop the retained turns now rather
                # than letting them sit until the TTL. The client clears its
                # transcript in the same gesture, so leaving the server half of
                # the conversation alive would be both surprising and needless
                # retention of what the previous user said.
                if _CARRYOVER["history"]:
                    log.info("new session: discarding %d carried-over messages",
                             len(_CARRYOVER["history"]))
                _CARRYOVER["history"] = []
                _CARRYOVER["at"] = 0.0
            await self.brain.start()
        except Exception as exc:
            log.exception("brain.start failed")
            self.enqueue({"type": "error", "code": "brain_start",
                          "text": f"Conversation brain failed to start: "
                                  f"{type(exc).__name__}: {exc}"})
            return
        log.info("brain started in %.0f ms", (time.perf_counter() - t0) * 1000.0)

    async def primer(self) -> None:
        """Wait for the first webcam frame, prime the engine, announce ready."""
        await self.configured.wait()        # the reference must be swapped first
        while not self.have_first_frame:
            if self.closing.is_set():
                return
            await asyncio.sleep(0.02)
        log.info("priming engine with the worker's first cropped frame")
        tic = time.perf_counter()
        try:
            await self.loop.run_in_executor(_GPU_EXECUTOR, self.gpu.prime)
        except LeaseError as exc:
            self._on_lease_lost(exc)
            return
        log.info("engine primed in %.0f ms", (time.perf_counter() - tic) * 1000.0)
        self._last_prime_at = time.monotonic()
        # The 50 primed frames are intentionally discarded: the client shows the
        # reference photo until the first generated block arrives.
        self.enqueue({"type": "ready", "fps": FPS, "sr": SR,
                      "block_frames": BLOCK_FRAMES, "block_samples": BLOCK_SAMPLES})
        self.primed.set()

    def stats(self) -> dict:
        n = max(1, self.block_idx)
        return {
            "blocks": self.block_idx,
            "uptime_s": round(time.monotonic() - self.t_start, 1),
            "mic_chunks": self.n_mic_chunks,
            "cam_frames": self.n_cam_frames,
            "cropped": self.n_cropped,
            "dropped_video": self.dropped_video,
            "avg_step_ms": round(self.step_ms_total / n, 1),
            "last_gpu_step_ms": round(self.gpu_step_ms, 1),
            "late_blocks": self.late_blocks,
            "reprimes": self.n_reprimes,
            "resyncs": self.n_resyncs,
            # Sampled at the last pacing log rather than fetched per request, so
            # /healthz never queues an RPC behind a 400 ms step.
            "pose_dist": round(self._engine_stats.get("pose_dist", 0.0), 2),
            "pose_deadband": round(self._engine_stats.get("pose_deadband", 0.0), 2),
            "pose_corrections": self._engine_stats.get("pose_corrections", 0),
            "brain_state": getattr(self.brain, "state", None),
            "lease_lost": self.lease_lost,
            # booleans only: the values themselves never leave the session
            "custom_reference": self.ref_applied,
            "custom_voice": "voice_id" in self.session_cfg,
            "custom_prompt": "system_prompt" in self.session_cfg,
            "resumed_messages": self.resumed_messages,
        }


# =========================================================================== #
# build_app
# =========================================================================== #
def build_app(
    engine_factory: Callable[[], Any],
    brain_factory: Callable[[Callable[[dict], None], dict], Any],
    face_cropper_factory: Callable[[], Any] | None = None,
    *,
    static_dir: str | None = None,
    preload: bool = True,
    default_system_prompt: str = "",
):
    """Create the gradio.Server app.

    Parameters
    ----------
    engine_factory        : () -> AvatarEngine-like (``warm``, ``start_session``,
                            ``step``, ``set_reference``). NOT called here: it is
                            handed to the GPU worker and called inside the lease.
    brain_factory         : (on_event, session_cfg) -> ConversationBrain-like.
                            ``session_cfg`` is the client's per-session config
                            dict; today only ``{"voice_id": str}`` when the user
                            supplied one (absent = use the server default).
    face_cropper_factory  : () -> FaceCropper-like (``crop``); optional. Also
                            constructed inside the lease — it touches CUDA.
    static_dir            : directory holding ``index.html`` (default ``../static``)
    preload               : unused on ZeroGPU (kept for signature compatibility);
                            there is nothing to preload without a GPU.
    default_system_prompt : the brain's built-in prompt, shipped to the client in
                            ``hello`` so its editor opens on the real default.
                            Passed in (not imported) to keep this module clean of
                            both the real brain and the mocks.
    """
    import spaces
    from gradio import Server  # imported here so the module is importable w/o gradio

    static = static_dir or STATIC_DIR
    app = Server(title="Real-Time Conversational Avatar", docs_url=None, redoc_url=None)

    holder = _LeaseHolder()
    state: dict[str, Any] = {"session": None, "sessions_total": 0, "lease": False}

    # ---------------------------------------------------------------- lease -- #
    # Registered as a Gradio API endpoint rather than a raw route on purpose:
    # the Gradio JS client performs the `zerogpu-headers` postMessage handshake
    # with the huggingface.co parent frame and attaches the visitor's
    # X-IP-Token, so the GPU seconds are billed to whoever is watching instead
    # of falling back to the Space's shared IP quota.
    @spaces.GPU(duration=gpu_session.LEASE_SECONDS, size=gpu_session.GPU_SIZE)
    def _hold_lease():
        yield from gpu_session.gpu_worker_body(
            engine_factory, face_cropper_factory, gpu_session.LEASE_SECONDS)

    @app.api(name="run_session")
    def run_session() -> str:
        """Hold one GPU lease for the duration of a conversation.

        Yields JSON status lines: ``ready`` (the client may now open ``/ws``),
        then a per-second ``tick`` countdown, then ``expired``.
        """
        if state["lease"]:
            yield json.dumps({"event": "busy",
                              "text": "Another session is running. Try again shortly."})
            return
        state["lease"] = True
        gpu_session.PROXY.open()
        try:
            for msg in _hold_lease():
                if msg.get("event") == "ready":
                    holder.on_ready(msg.get("warm_seconds", 0.0),
                                    msg.get("session_seconds", 0.0))
                yield json.dumps(msg)
        except Exception as exc:
            log.exception("lease failed")
            holder.load_error = f"{type(exc).__name__}: {exc}"
            yield json.dumps({"event": "error", "text": holder.load_error})
        finally:
            gpu_session.PROXY.close("the GPU lease ended")
            state["lease"] = False

    # ---- routes (registered BEFORE launch() so they win over gradio's) ---- #
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        path = os.path.join(static, "index.html")
        if not os.path.exists(path):
            return HTMLResponse("<h1>static/index.html missing</h1>", status_code=500)
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.get("/defaults")
    async def defaults() -> JSONResponse:
        """Server-side defaults the UI needs *before* any websocket exists.

        The prompt editor opens on the real prompt this way, instead of on a
        blank box the user would have to guess at. Same payload as the ``hello``
        fields, so either source seeds the editor.
        """
        return JSONResponse({
            "system_prompt": default_system_prompt,
            "max_prompt_chars": MAX_PROMPT_CHARS,
        })

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        sess = state["session"]
        return JSONResponse({
            "ok": True,
            "lease_held": holder.loaded,
            "lease_warm_seconds": holder.warm_seconds,
            "lease_session_seconds": holder.session_seconds,
            "lease_error": holder.load_error,
            "gpu_size": gpu_session.GPU_SIZE,
            "lease_seconds": gpu_session.LEASE_SECONDS,
            "busy": sess is not None,
            "sessions_total": state["sessions_total"],
            "session": sess.stats() if sess is not None else None,
        })

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        # ---- single concurrent session ---- #
        if state["session"] is not None:
            await ws.send_text(json.dumps({
                "type": "error", "code": "busy",
                "text": "Another session is already running. Try again in a moment.",
            }))
            with contextlib.suppress(Exception):
                await ws.close(code=1013)  # try again later
            return

        # The client is supposed to hold a lease (via /run_session) before it
        # opens this socket, so this is a real error rather than a wait.
        try:
            gpu = await holder.get()
        except Exception as exc:
            with contextlib.suppress(Exception):
                await ws.send_text(json.dumps({
                    "type": "error", "code": "no_lease",
                    "text": f"No GPU session: {exc}"}))
                await ws.close(code=1011)
            return

        # A new user is (probably) framed differently: drop the previous session's
        # EMA face box so the first detection re-locks immediately instead of
        # inheriting a stale crop for up to `detect_every` frames.
        with contextlib.suppress(Exception):
            await asyncio.get_running_loop().run_in_executor(
                _CROP_EXECUTOR, gpu.reset_cropper)

        session = _Session(ws, gpu, brain_factory)
        state["session"] = session
        state["sessions_total"] += 1
        log.info("session #%d started", state["sessions_total"])

        tasks: list[asyncio.Task] = []
        try:
            session.enqueue({"type": "hello", "server": "avatar", "fps": FPS,
                             "sr": SR, "block_frames": BLOCK_FRAMES,
                             "system_prompt": default_system_prompt,
                             "max_prompt_chars": MAX_PROMPT_CHARS})
            tasks = [
                asyncio.create_task(session.writer(), name="ws-writer"),
                asyncio.create_task(session.reader(), name="ws-reader"),
                asyncio.create_task(session.configurator(), name="configurator"),
                asyncio.create_task(session.brain_starter(), name="brain-starter"),
                asyncio.create_task(session.primer(), name="primer"),
                asyncio.create_task(session.generator(), name="generator"),
                asyncio.create_task(session.closer(), name="closer"),
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:  # surface real errors; a disconnect or an expired
                exc = t.exception()   # lease are both ordinary endings
                if exc and not isinstance(exc, (WebSocketDisconnect, LeaseError)):
                    log.error("task %s died: %r", t.get_name(), exc)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("session error")
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            if session.brain is not None:
                # Save the turn list before stopping, so the client's "Keep
                # going" can pick the conversation up on the next lease.
                if hasattr(session.brain, "export_history"):
                    with contextlib.suppress(Exception):
                        hist = session.brain.export_history()
                        if hist:
                            _CARRYOVER["history"] = hist
                            _CARRYOVER["at"] = time.monotonic()
                with contextlib.suppress(Exception):
                    await session.brain.stop()
            # No leakage: the next user gets the default face back. Only worth
            # doing while the lease is still alive -- a dead worker has no state
            # left to leak, and the next lease rebuilds from the default anyway.
            if session.ref_applied and gpu.live:
                with contextlib.suppress(Exception):
                    await asyncio.get_running_loop().run_in_executor(
                        _GPU_EXECUTOR, gpu.set_reference, None)
            log.info("session #%d ended: %s", state["sessions_total"], session.stats())
            state["session"] = None
            with contextlib.suppress(Exception):
                await ws.close()

    # NOTE: the dedicated Space preloaded the engine here (a middleware kick,
    # because gradio's own lifespan swallows @app.on_event("startup")). On
    # ZeroGPU there is nothing to preload: the weights are already built and
    # packed at import in app.py, and everything else needs the lease.
    app.state.lease_holder = holder      # handy for tests / integration
    app.preload_blocking = lambda *a, **k: None  # type: ignore[attr-defined]
    return app

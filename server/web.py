"""Web layer: gradio.Server app + WebSocket real-time A/V session loop.

``build_app(engine_factory, brain_factory, face_cropper_factory)`` returns a
``gradio.Server`` instance (a FastAPI subclass) with:

    GET  /            -> static/index.html
    GET  /defaults    -> JSON {system_prompt, max_prompt_chars} for the UI editor
    GET  /healthz     -> JSON status (engine loaded? session busy?)
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
  * ``_GPU_EXECUTOR``  – ONE thread, module-global. Every ``engine.load()``,
    ``engine.start_session()`` and ``engine.step()`` runs there, so the GPU is
    never re-entered concurrently even across reconnects.
  * ``_CROP_EXECUTOR`` – ONE thread. JPEG decode + ``FaceCropper.crop`` (the
    real cropper keeps EMA state, so it must stay single-threaded).
  * ``_JPEG_EXECUTOR`` – small pool for outbound JPEG encoding (Pillow releases
    the GIL in the encoder).
  * a single writer task owns the WebSocket; every other task enqueues frames,
    so we never interleave two concurrent ``send_bytes`` calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

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
# Frames of cross-dissolve used to hide the pose cut of a drift re-prime (0 = hard cut).
REPRIME_XFADE = int(os.environ.get("AVATAR_REPRIME_XFADE", "10"))
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
OUT_JPEG_QUALITY = int(os.environ.get("AVATAR_JPEG_QUALITY", "72"))
OUT_JPEG_SUBSAMPLING = int(os.environ.get("AVATAR_JPEG_SUBSAMPLING", "2"))  # 2 = 4:2:0
SEND_QUEUE_MAX = 240            # ~9 blocks of video+audio; video dropped if full
# Side of the square webcam frame the frontend sends (index.html: CAM_SIZE).
# Used only to warm the face detector at load time -- see _warm_cropper().
CAM_WARM_SIZE = int(os.environ.get("AVATAR_CAM_WARM_SIZE", "384"))
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
# create a second GPU thread.
_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
_CROP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crop")
_JPEG_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jpeg")


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


def _jpeg_bytes_to_rgb(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _rgb_to_jpeg(arr: np.ndarray, quality: int = OUT_JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality,
                                     subsampling=OUT_JPEG_SUBSAMPLING, optimize=False)
    return buf.getvalue()


def _xfade(prev: np.ndarray | None, new: np.ndarray) -> np.ndarray:
    """Cross-dissolve ``new`` in from the still frame ``prev``.

    A drift re-prime restarts the rollout from the avatar's reference state, so
    the pose can jump. Dissolving over the block (400 ms) turns that cut into a
    soft morph, which on an idle talking head is essentially invisible. ~6 ms of
    numpy for 10 frames of 512x512.
    """
    if prev is None or REPRIME_XFADE <= 0:
        return new
    out = np.array(new, dtype=np.uint8, copy=True)
    k = min(REPRIME_XFADE, len(out))
    base = prev.astype(np.float32)
    for i in range(k):
        a = (i + 1) / (k + 1)          # 0 < a < 1: frame 0 stays close to `prev`
        out[i] = (base * (1.0 - a) + out[i].astype(np.float32) * a).astype(np.uint8)
    return out


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
# lazily-constructed, process-wide engine + cropper
# =========================================================================== #
class _EngineHolder:
    """Builds and ``load()``s the engine exactly once, inside the GPU thread.

    The mutual exclusion is a *threading* lock (not ``asyncio.Lock``) so the
    heavy load can also be driven synchronously from ``__main__`` *before*
    ``app.launch()`` — ``engine.load()`` blocks ~36 s for the real engine and
    ``server/engine.py``'s caveat (a) asks for it to happen before any
    websocket is accepted. See :meth:`preload_blocking`.
    """

    def __init__(self, engine_factory: Callable[[], Any],
                 cropper_factory: Callable[[], Any] | None) -> None:
        self._engine_factory = engine_factory
        self._cropper_factory = cropper_factory
        self._engine: Any = None
        self._cropper: Any = None
        self._lock = threading.Lock()
        self.load_error: str | None = None
        self.load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    # -- runs *in* the GPU thread (so the CUDA context belongs to it) -------- #
    def _ensure_sync(self) -> tuple[Any, Any]:
        with self._lock:
            if self._engine is not None:
                return self._engine, self._cropper
            t0 = time.perf_counter()
            try:
                eng = self._engine_factory()
                eng.load()
                self._engine = eng
                if self._cropper_factory is not None:
                    self._cropper = _CROP_EXECUTOR.submit(self._cropper_factory).result()
                    _CROP_EXECUTOR.submit(self._warm_cropper).result()
                self.load_seconds = time.perf_counter() - t0
                self.load_error = None
                log.info("engine loaded in %.2fs", self.load_seconds)
            except Exception as exc:  # keep the server alive, report to client
                self.load_error = f"{type(exc).__name__}: {exc}"
                log.exception("engine load failed")
                raise
            return self._engine, self._cropper

    def _warm_cropper(self) -> None:
        """Run one crop at the frontend's frame size, on the crop thread.

        Why this matters: ``face_alignment``'s SFD detector sets
        ``cudnn.benchmark = True`` on every call, so the *first* detection at a
        given input shape pays a full cuDNN autotune. The engine's own warm-up
        uses a 360x480 dummy, but ``index.html`` always sends a **square** frame
        (CAM_SIZE), which the cropper rescales to 360x360 -- a different shape,
        hence a second autotune. Measured cost of that autotune: **15.6 s**, and
        without this warm-up it lands on the first webcam frame of the first
        session, delaying ``{"type":"ready"}`` by exactly that much.
        Any square client frame maps to 360x360, so one warm-up covers them all.
        """
        if self._cropper is None:
            return
        t0 = time.perf_counter()
        try:
            rng = np.random.default_rng(0)
            dummy = rng.integers(0, 255, (CAM_WARM_SIZE, CAM_WARM_SIZE, 3),
                                 dtype=np.uint8)
            self._cropper.crop(dummy)
            if hasattr(self._cropper, "reset"):
                self._cropper.reset()
        except Exception:
            log.exception("cropper warm-up failed (non-fatal)")
            return
        log.info("cropper warmed at %dx%d in %.0f ms",
                 CAM_WARM_SIZE, CAM_WARM_SIZE, (time.perf_counter() - t0) * 1000.0)

    def preload_blocking(self, timeout: float | None = None) -> None:
        """Build + load the engine now, from a synchronous context."""
        _GPU_EXECUTOR.submit(self._ensure_sync).result(timeout)

    async def get(self) -> tuple[Any, Any]:
        if self._engine is not None:          # fast path, no executor hop
            return self._engine, self._cropper
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_GPU_EXECUTOR, self._ensure_sync)


# =========================================================================== #
# one WebSocket session
# =========================================================================== #
class _Session:
    def __init__(self, ws: WebSocket, engine: Any, cropper: Any,
                 brain_factory: Callable[..., Any]) -> None:
        self.ws = ws
        self.engine = engine
        self.cropper = cropper
        # Built by configurator(), not here: the client's voice_id has to reach
        # brain_factory, and it only arrives with `session_config`.
        self.brain_factory = brain_factory
        self.brain: Any = None
        self.loop = asyncio.get_running_loop()

        self.mic = MicRing()
        self.frames: list[np.ndarray] = []     # last <=10 cropped 512 RGB frames
        self.latest_jpeg: bytes | None = None  # newest raw webcam JPEG
        self.first_frame: np.ndarray | None = None
        self.primed = asyncio.Event()
        self.closing = asyncio.Event()

        # per-session config handshake (see the protocol docstring)
        self.session_cfg: dict[str, Any] = {}  # holds voice_id; never logged
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
        self.n_mic_chunks = 0
        self.n_cam_frames = 0
        self.n_cropped = 0
        self.dropped_video = 0
        self.step_ms_total = 0.0
        self.late_blocks = 0
        self.n_reprimes = 0
        self._last_prime_at = 0.0
        self._last_out_frame: np.ndarray | None = None   # for the re-prime dissolve

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
        """Decode + face-crop the newest webcam JPEG (one at a time)."""
        try:
            def _work(buf: bytes) -> np.ndarray:
                rgb = _jpeg_bytes_to_rgb(buf)
                return self.cropper.crop(rgb) if self.cropper is not None else rgb

            cropped = await self.loop.run_in_executor(_CROP_EXECUTOR, _work, jpeg)
            self.frames.append(cropped)
            if len(self.frames) > BLOCK_FRAMES:
                del self.frames[: len(self.frames) - BLOCK_FRAMES]
            self.n_cropped += 1
            if self.first_frame is None:
                self.first_frame = cropped
        except Exception:
            log.exception("crop failed")
        finally:
            self._crop_busy = False

    # ---------------- generation ---------------- #
    def _user_frames(self) -> np.ndarray:
        """Last 10 cropped frames, repeating the newest if we have fewer."""
        have = list(self.frames)
        if not have:
            size = getattr(self.engine, "SIZE", 512)
            have = [np.zeros((size, size, 3), dtype=np.uint8)]
        while len(have) < BLOCK_FRAMES:
            have.insert(0, have[0])
        return np.stack(have[-BLOCK_FRAMES:], axis=0)

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
            user_frames = self._user_frames()

            tic = time.perf_counter()
            if self._want_reprime(avatar_pcm16):
                # Drift reset: a fresh start_session() returns the 50 priming
                # frames; the last 10 are this block's output, so the 400 ms
                # cadence and the block indices are unaffected.
                primed = await self.loop.run_in_executor(
                    _GPU_EXECUTOR, self.engine.start_session, user_frames[-1])
                frames = _xfade(self._last_out_frame, primed[-BLOCK_FRAMES:])
                self._last_prime_at = time.monotonic()
                self.n_reprimes += 1
                log.info("re-primed engine at block %d (drift reset #%d)",
                         b, self.n_reprimes)
            else:
                frames = await self.loop.run_in_executor(
                    _GPU_EXECUTOR, self.engine.step, avatar_f32, user_f32, user_frames)
            step_ms = (time.perf_counter() - tic) * 1000.0
            if len(frames):
                self._last_out_frame = frames[-1]
            self.step_ms_total += step_ms
            self._log_step_ms += step_ms
            self.step_ms_max = max(self.step_ms_max, step_ms)

            jpegs = await asyncio.gather(*[
                self.loop.run_in_executor(_JPEG_EXECUTOR, _rgb_to_jpeg, frames[i])
                for i in range(min(BLOCK_FRAMES, len(frames)))
            ])
            for i, jpg in enumerate(jpegs):
                self.enqueue(_pack_video(b, i, jpg), droppable=True)

            if time.monotonic() - (t0 + (b + 1) * BLOCK_SEC) > 0.2:
                self.late_blocks += 1

            # periodic pacing telemetry (integration/debug: see reports/integration.md)
            n = self.block_idx - self._log_at
            if PACING_LOG_BLOCKS and n >= PACING_LOG_BLOCKS:
                log.info(
                    "pacing: blocks=%d step_ms avg=%.0f (window) max=%.0f  late=%d "
                    "dropped_video=%d sendq=%d reprimes=%d pose_d=%.1f/%.1f "
                    "pose_corr=%d brain=%s",
                    self.block_idx, self._log_step_ms / n, self.step_ms_max,
                    self.late_blocks, self.dropped_video, self.send_q.qsize(),
                    self.n_reprimes, getattr(self.engine, "pose_dist", 0.0),
                    getattr(self.engine, "_pose_deadband", 0.0),
                    getattr(self.engine, "n_pose_corr", 0),
                    getattr(self.brain, "state", "?"),
                )
                self._log_at = self.block_idx
                self._log_step_ms = 0.0
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

        has_set = hasattr(self.engine, "set_reference")
        try:
            if self.ref_bytes is not None and has_set:
                secs = await self.loop.run_in_executor(
                    _GPU_EXECUTOR, self.engine.set_reference, self.ref_bytes)
                self.ref_applied = True
                log.info("custom reference applied (%.0f KB) in %.0f ms",
                         len(self.ref_bytes) / 1024.0, secs * 1000.0)
                self.enqueue({"type": "reference", "ok": True})
            elif has_set and not getattr(self.engine, "reference_is_default", True):
                # Previous session uploaded one and died before restoring it.
                secs = await self.loop.run_in_executor(
                    _GPU_EXECUTOR, self.engine.set_reference, None)
                log.info("restored default reference in %.0f ms", secs * 1000.0)
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
        while self.first_frame is None:
            await asyncio.sleep(0.02)
        first = self.first_frame
        log.info("priming engine with first cropped frame %s", first.shape)
        tic = time.perf_counter()
        await self.loop.run_in_executor(_GPU_EXECUTOR, self.engine.start_session, first)
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
            "late_blocks": self.late_blocks,
            "reprimes": self.n_reprimes,
            "pose_dist": round(getattr(self.engine, "pose_dist", 0.0), 2),
            "pose_deadband": round(getattr(self.engine, "_pose_deadband", 0.0), 2),
            "pose_corrections": getattr(self.engine, "n_pose_corr", 0),
            "brain_state": getattr(self.brain, "state", None),
            # booleans only: the values themselves never leave the session
            "custom_reference": self.ref_applied,
            "custom_voice": "voice_id" in self.session_cfg,
            "custom_prompt": "system_prompt" in self.session_cfg,
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
    engine_factory        : () -> AvatarEngine-like (``load``, ``start_session``,
                            ``step``, and optionally ``set_reference`` +
                            ``reference_is_default`` for user-supplied photos)
    brain_factory         : (on_event, session_cfg) -> ConversationBrain-like.
                            ``session_cfg`` is the client's per-session config
                            dict; today only ``{"voice_id": str}`` when the user
                            supplied one (absent = use the server default).
    face_cropper_factory  : () -> FaceCropper-like (``crop``); optional
    static_dir            : directory holding ``index.html`` (default ``../static``)
    preload               : build+load the engine at server startup
    default_system_prompt : the brain's built-in prompt, shipped to the client in
                            ``hello`` so its editor opens on the real default.
                            Passed in (not imported) to keep this module clean of
                            both the real brain and the mocks.
    """
    from gradio import Server  # imported here so the module is importable w/o gradio

    static = static_dir or STATIC_DIR
    app = Server(title="Real-Time Conversational Avatar", docs_url=None, redoc_url=None)

    holder = _EngineHolder(engine_factory, face_cropper_factory)
    state: dict[str, Any] = {"session": None, "sessions_total": 0}

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
            "engine_loaded": holder.loaded,
            "engine_load_seconds": holder.load_seconds,
            "engine_load_error": holder.load_error,
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

        try:
            engine, cropper = await holder.get()
        except Exception:
            with contextlib.suppress(Exception):
                await ws.send_text(json.dumps({
                    "type": "error", "code": "engine_load",
                    "text": f"Engine failed to load: {holder.load_error}"}))
                await ws.close(code=1011)
            return

        # A new user is (probably) framed differently: drop the previous session's
        # EMA face box so the first detection re-locks immediately instead of
        # inheriting a stale crop for up to `detect_every` frames.
        if cropper is not None and hasattr(cropper, "reset"):
            with contextlib.suppress(Exception):
                await asyncio.get_running_loop().run_in_executor(
                    _CROP_EXECUTOR, cropper.reset)

        session = _Session(ws, engine, cropper, brain_factory)
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
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:  # surface non-disconnect errors in the log
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
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
                with contextlib.suppress(Exception):
                    await session.brain.stop()
            # No leakage: the next user gets the default face back. Paid here so
            # a plain session never waits for it at connect time.
            if session.ref_applied and hasattr(engine, "set_reference"):
                with contextlib.suppress(Exception):
                    await asyncio.get_running_loop().run_in_executor(
                        _GPU_EXECUTOR, engine.set_reference, None)
            log.info("session #%d ended: %s", state["sessions_total"], session.stats())
            state["session"] = None
            with contextlib.suppress(Exception):
                await ws.close()

    if preload:
        # NOTE: @app.on_event("startup") / router.on_startup are IGNORED here --
        # gradio's App supplies its own `lifespan`, so FastAPI never runs them.
        # Instead we kick the (idempotent) load off on the first HTTP request;
        # gradio's own launch-time HEAD "/" makes that happen immediately.
        @app.middleware("http")
        async def _preload_kick(request, call_next):
            if not state.get("preload_started"):
                state["preload_started"] = True

                async def _go() -> None:
                    with contextlib.suppress(Exception):
                        await holder.get()

                asyncio.create_task(_go())
            return await call_next(request)

    app.state.engine_holder = holder  # handy for tests / integration
    # Call this from __main__ *before* app.launch() to pay the ~36 s model load
    # up front instead of inside the first websocket connect.
    app.preload_blocking = holder.preload_blocking  # type: ignore[attr-defined]
    return app

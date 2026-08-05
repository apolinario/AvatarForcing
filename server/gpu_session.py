"""ZeroGPU held-lease GPU worker + parent-side RPC proxy.

ZeroGPU imposes two constraints: there is no GPU outside ``@spaces.GPU``, and
every ``@spaces.GPU`` call runs in a separate worker process, so per-call
decorators would lose the engine's KV cache between blocks. The GPU is
therefore leased once per conversation: ``gpu_worker_body`` is the body of a
``@spaces.GPU`` generator that warms the engine, then serves an RPC loop over
fork-context queues until the lease expires.

The worker owns the face cropper, the cropped-frame ring and JPEG encoding, so
the wire carries ~25 KB of webcam JPEG in and ~200 KB of encoded frames out
per 400 ms block instead of ~16 MB of raw frames.

The worker runs three interchangeable dispatch lanes so a 400 ms ``step``, a
~20 ms crop and a ~740 ms speech synthesis never head-of-line block each other
in the queue (they still serialise on the GPU). Threads only: a ``@spaces.GPU``
fork is daemonic and cannot spawn child processes.

The worker also hosts OmniVoice TTS and Whisper STT (``server/speech.py``) on
the same leased GPU.
"""

from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing as _mp
import os
import queue as _queue
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

log = logging.getLogger("avatar.gpu")

# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
# Seconds of conversation the visitor gets. The clock starts when the engine is
# warm, not when the GPU is granted, so warm-up never eats conversation time.
SESSION_SECONDS = int(os.environ.get("AVATAR_SESSION_SECONDS", "90"))
# Headroom for that warm-up: engine warm (4-13 s) plus the live ASR model,
# which cannot load before the lease (server/speech.py).
WARM_ALLOWANCE = float(os.environ.get("AVATAR_WARM_ALLOWANCE", "25.0"))
# Stop serving this many seconds early so the worker tears down cleanly.
LEASE_MARGIN = float(os.environ.get("AVATAR_LEASE_MARGIN", "3.0"))
# What @spaces.GPU(duration=...) requests. Longer leases queue lower and weigh
# more against the visitor's quota, so no more slack than warm-up needs.
LEASE_SECONDS = int(SESSION_SECONDS + WARM_ALLOWANCE + LEASE_MARGIN)
# "large" is half the card (48 GB, 1x quota): the full pipeline steps in
# ~150-230 ms against a 400 ms block budget, so "xlarge" would double the
# visitor's quota cost for headroom that is already there.
GPU_SIZE = os.environ.get("AVATAR_GPU_SIZE", "large")
# How long a parent-side RPC waits before giving up on the worker.
CALL_TIMEOUT = float(os.environ.get("AVATAR_CALL_TIMEOUT", "120.0"))

BLOCK_FRAMES = 10
OUT_JPEG_QUALITY = int(os.environ.get("AVATAR_JPEG_QUALITY", "72"))
OUT_JPEG_SUBSAMPLING = int(os.environ.get("AVATAR_JPEG_SUBSAMPLING", "2"))
REPRIME_XFADE = int(os.environ.get("AVATAR_REPRIME_XFADE", "10"))

# --------------------------------------------------------------------------- #
# fork-context queues, one pair per concurrent session, ALL built at import.
#
# ZeroGPU reuses worker processes, and a reused worker only has the module
# globals from its original fork -- a queue created later is invisible to it.
# A pool that predates every fork is visible to all of them; sessions claim a
# slot and pass its index (an int pickles, a Queue does not).
# --------------------------------------------------------------------------- #
_CTX = _mp.get_context("fork")

# Concurrent conversations. Each holds its own worker with its own restored
# copy of the weights, so the bound is VRAM, not the ZeroGPU scheduler.
MAX_SESSIONS = int(os.environ.get("AVATAR_MAX_SESSIONS", "4"))

LANES: list[tuple[Any, Any]] = [
    (_CTX.Queue(),   # parent -> worker: (lease_id, seq, lane, method, args)
     _CTX.Queue())   # worker -> parent: (seq, ok, payload)
    for _ in range(MAX_SESSIONS)]

_SLOT_LOCK = threading.Lock()
_SLOTS_FREE: list[int] = list(range(MAX_SESSIONS))


def claim_slot() -> int | None:
    """Take a free session slot, or None when the Space is at capacity."""
    with _SLOT_LOCK:
        return _SLOTS_FREE.pop(0) if _SLOTS_FREE else None


def release_slot(slot: int) -> None:
    with _SLOT_LOCK:
        if slot not in _SLOTS_FREE:
            _SLOTS_FREE.append(slot)
            _SLOTS_FREE.sort()


def slots_in_use() -> int:
    with _SLOT_LOCK:
        return MAX_SESSIONS - len(_SLOTS_FREE)

_STOP = "__stop__"
_LEASE_GONE = "__lease_gone__"     # distinguishes "worker vanished" from "call raised"
LANE_ENGINE = "engine"
LANE_CROP = "crop"

# Which lease each slot's parent is talking to. Stamped on every request so a
# worker can reject calls left in the queue by a dead session, and passed as a
# call argument rather than read from a module global in the worker (a reused
# worker's globals are stale). Per slot: sessions are independent.
LEASE_IDS: list[int] = [0] * MAX_SESSIONS
_LEASE_LOCK = threading.Lock()


# =========================================================================== #
# worker side (runs inside the fork, with a real GPU)
# =========================================================================== #
class _Worker:
    """Owns the engine, the cropper, the frame ring, JPEG encoding and speech."""

    def __init__(self, engine, cropper, speech=None) -> None:
        self.engine = engine
        self.cropper = cropper
        # OmniVoice TTS + Whisper STT. Shares the GPU with the avatar engine;
        # see server/speech.py for the budget this has to fit in.
        self.speech = speech
        self.frames: list[np.ndarray] = []
        self.first_frame: np.ndarray | None = None
        self.n_cropped = 0
        self._lock = threading.Lock()
        self._jpeg = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jpeg")
        self._last_out: np.ndarray | None = None

    # -- helpers ---------------------------------------------------------- #
    def _encode(self, frames) -> list[bytes]:
        def one(arr: np.ndarray) -> bytes:
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(arr, "RGB").save(
                buf, format="JPEG", quality=OUT_JPEG_QUALITY,
                subsampling=OUT_JPEG_SUBSAMPLING, optimize=False)
            return buf.getvalue()

        n = min(BLOCK_FRAMES, len(frames))
        return list(self._jpeg.map(one, [frames[i] for i in range(n)]))

    def _user_frames(self) -> np.ndarray:
        """Last 10 cropped frames, repeating the oldest if we have fewer."""
        with self._lock:
            have = list(self.frames)
        if not have:
            size = getattr(self.engine, "SIZE", 512)
            have = [np.zeros((size, size, 3), dtype=np.uint8)]
        while len(have) < BLOCK_FRAMES:
            have.insert(0, have[0])
        return np.stack(have[-BLOCK_FRAMES:], axis=0)

    # -- RPC methods ------------------------------------------------------ #
    def push_frame(self, jpeg: bytes) -> dict:
        """Decode + face-crop one webcam frame into the ring (crop lane)."""
        from PIL import Image

        with Image.open(io.BytesIO(jpeg)) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        cropped = self.cropper.crop(rgb) if self.cropper is not None else rgb
        with self._lock:
            self.frames.append(cropped)
            if len(self.frames) > BLOCK_FRAMES:
                del self.frames[: len(self.frames) - BLOCK_FRAMES]
            if self.first_frame is None:
                self.first_frame = cropped
            self.n_cropped += 1
            n = self.n_cropped
        return {"n_cropped": n, "have_first": True}

    def prime(self) -> dict:
        """``start_session`` on the first cropped frame. Primed frames are
        discarded; the client shows the reference photo until the first
        generated block lands."""
        with self._lock:
            first = self.first_frame
        if first is None:
            raise RuntimeError("prime() before any webcam frame")
        t0 = time.perf_counter()
        self.engine.start_session(first)
        return {"ms": (time.perf_counter() - t0) * 1000.0}

    def step(self, avatar_f32: np.ndarray, user_f32: np.ndarray) -> dict:
        t0 = time.perf_counter()
        frames = self.engine.step(avatar_f32, user_f32, self._user_frames())
        step_ms = (time.perf_counter() - t0) * 1000.0
        if len(frames):
            self._last_out = frames[-1]
        return {"jpegs": self._encode(frames), "step_ms": step_ms}

    def reprime(self) -> dict:
        """Drift reset: re-prime and cross-dissolve into the new pose."""
        t0 = time.perf_counter()
        primed = self.engine.start_session(self._user_frames()[-1])
        new = primed[-BLOCK_FRAMES:]
        prev, self._last_out = self._last_out, (new[-1] if len(new) else None)
        if prev is not None and REPRIME_XFADE > 0:
            out = np.array(new, dtype=np.uint8, copy=True)
            k = min(REPRIME_XFADE, len(out))
            base = prev.astype(np.float32)
            for i in range(k):
                a = (i + 1) / (k + 1)
                out[i] = (base * (1.0 - a)
                          + out[i].astype(np.float32) * a).astype(np.uint8)
            new = out
        return {"jpegs": self._encode(new),
                "step_ms": (time.perf_counter() - t0) * 1000.0}

    def set_reference(self, image: bytes | None) -> dict:
        secs = self.engine.set_reference(image)
        return {"secs": secs,
                "is_default": bool(getattr(self.engine, "reference_is_default", True))}

    def reset_cropper(self) -> dict:
        with self._lock:
            self.frames.clear()
            self.first_frame = None
        if self.cropper is not None and hasattr(self.cropper, "reset"):
            self.cropper.reset()
        return {}

    # -- speech RPC ------------------------------------------------------- #
    def tts_set_voice(self, ref_bytes, ref_text, key) -> dict:
        return self.speech.set_voice(ref_bytes, ref_text, key)

    def tts_synth(self, text: str) -> dict:
        return self.speech.synth(text)

    def stt_transcribe(self, pcm: bytes, sr: int) -> dict:
        return self.speech.transcribe(pcm, sr)

    def stats(self) -> dict:
        return {
            "pose_dist": float(getattr(self.engine, "pose_dist", 0.0)),
            "pose_deadband": float(getattr(self.engine, "_pose_deadband", 0.0)),
            "pose_corrections": int(getattr(self.engine, "n_pose_corr", 0)),
            "n_cropped": self.n_cropped,
        }


def _serve(worker: _Worker, deadline: float, stop: threading.Event,
           my_lease: int, slot: int) -> None:
    """Pull work off this slot's queues until stopped or expired.

    Lanes are interchangeable pullers, not routed queues: they exist so a
    ~20 ms crop or a speech call never waits behind a ~400 ms ``step`` in the
    queue. web.py keeps at most one crop and one step in flight and speech
    adds a third caller, so three lanes suffice.
    """
    req_q, res_q = LANES[slot]
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            item = req_q.get(timeout=min(0.25, remaining))
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            return
        if item is None or item == _STOP:
            stop.set()                          # tell the sibling lane too
            return
        lease_id, seq, _lane, method, args = item
        if lease_id != my_lease:
            # A previous session's in-flight call. Answering it as a dead lease
            # lets that session finish dying instead of it corrupting this one.
            log.info("slot %d dropping stale request %s from lease %s (serving %s)",
                     slot, method, lease_id, my_lease)
            res_q.put((seq, _LEASE_GONE, "that GPU session has ended"))
            continue
        try:
            payload = getattr(worker, method)(*args)
            res_q.put((seq, True, payload))
        except Exception as exc:                # never kill the lane on one bad call
            log.exception("rpc %s failed", method)
            res_q.put((seq, False, f"{type(exc).__name__}: {exc}"))


def _pin_cudnn_benchmark_off() -> None:
    """Make ``cudnn.benchmark = True`` a no-op for the life of the worker.

    ``face_alignment``'s SFD detector re-enables it on every detection, and
    with benchmark on every new input shape pays a multi-second cuDNN autotune
    -- inside the visitor's session, since each lease is a fresh fork.
    Heuristic algorithm choice costs a little per conv and saves seconds.
    """
    import torch

    cudnn = torch.backends.cudnn
    try:
        type(cudnn).benchmark = property(lambda self: False,
                                         lambda self, value: None)
    except Exception:
        log.exception("could not pin cudnn.benchmark off (non-fatal)")
        return
    torch.backends.cudnn.deterministic = True


def gpu_worker_body(engine_factory, cropper_factory, lease_seconds: int,
                    speech_factory=None, lease_id: int = 0, slot: int = 0):
    """The body of the lease. A generator so the caller can stream progress.

    Yields a ``ready`` event once everything is warm, then a per-second
    ``tick`` with the remaining seconds, then ``expired``.
    """
    t_lease_start = time.monotonic()
    # Be done before ZeroGPU reclaims the process.
    hard_deadline = t_lease_start + lease_seconds - LEASE_MARGIN

    t0 = time.perf_counter()
    _pin_cudnn_benchmark_off()
    engine = engine_factory()
    # Weights were built and packed in the parent; this restores them to VRAM
    # and pays the reference-latent + cuDNN warm-up that needs a real GPU.
    engine.warm()
    cropper = cropper_factory() if cropper_factory is not None else None
    if cropper is not None:
        # Warm the square webcam-frame shape; the engine warm-up covers only
        # the reference shape.
        rng = np.random.default_rng(0)
        size = int(os.environ.get("AVATAR_CAM_WARM_SIZE", "384"))
        try:
            cropper.crop(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
            if hasattr(cropper, "reset"):
                cropper.reset()
        except Exception:
            log.exception("cropper warm-up failed (non-fatal)")
    warm_s = time.perf_counter() - t0
    log.info("engine warm in %.1fs", warm_s)

    speech = None
    if speech_factory is not None:
        t_s = time.perf_counter()
        try:
            speech = speech_factory()
            log.info("speech (OmniVoice + Whisper) ready in %.1fs",
                     time.perf_counter() - t_s)
        except Exception:
            log.exception("speech init failed -- the avatar will be mute")

    # Session clock starts after everything the first block needs, speech
    # included; warm-up runs on WARM_ALLOWANCE, not conversation time.
    deadline = min(time.monotonic() + SESSION_SECONDS, hard_deadline)
    log.info("lease up: warm %.1fs total, %.0fs of session",
             time.monotonic() - t_lease_start, deadline - time.monotonic())

    worker = _Worker(engine, cropper, speech)

    # Drain requests a previous, aborted session left behind.
    while True:
        try:
            LANES[slot][0].get_nowait()
        except _queue.Empty:
            break

    stop = threading.Event()
    lanes = [threading.Thread(target=_serve,
                              args=(worker, deadline, stop, lease_id, slot),
                              name=f"gpu-lane-{i}", daemon=True)
             for i in range(3)]
    for t in lanes:
        t.start()

    # The lanes must not outlive this generator: the process is reused, and a
    # lane still running after its lease ended would compete with the next
    # lease for the same queue. Early exit (visitor presses stop) is the
    # common path, hence the finally.
    try:
        yield {"event": "ready", "warm_seconds": round(warm_s, 1),
               "session_seconds": round(deadline - time.monotonic(), 1)}

        while any(t.is_alive() for t in lanes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
            yield {"event": "tick",
                   "remaining": round(max(0.0, deadline - time.monotonic()), 1)}
    finally:
        # Runs on normal expiry AND on GeneratorExit when the client cancels.
        stop.set()
        for t in lanes:
            t.join(timeout=2.0)
        still = [t.name for t in lanes if t.is_alive()]
        if still:
            log.warning("lanes did not stop: %s (a call is mid-flight)", still)

    log.info("lease over after %.0fs", time.monotonic() - t_lease_start)
    yield {"event": "expired"}


# =========================================================================== #
# parent side (no GPU, no CUDA -- just queues)
# =========================================================================== #
class LeaseError(RuntimeError):
    """No GPU lease is currently held, or it went away mid-call."""


class GPUProxy:
    """Parent-side handle on the worker. Thread-safe, multiplexed by sequence.

    ``web.py`` drives this from two threads (an engine executor and a crop
    executor), so responses cannot be matched by FIFO order. A single reader
    thread drains ``RES_Q`` into per-sequence slots instead.
    """

    def __init__(self, slot: int = 0) -> None:
        # One proxy per conversation, bound to its slot's queue pair.
        self.slot = slot
        self.req_q, self.res_q = LANES[slot]
        self._seq = 0
        self.lease_id = 0
        self._seq_lock = threading.Lock()
        self._pending: dict[int, list] = {}
        self._pending_lock = threading.Lock()
        self._cv = threading.Condition(self._pending_lock)
        self._alive = threading.Event()
        self._reader: threading.Thread | None = None
        self.reference_is_default = True
        self.last_error: str | None = None

    # -- lifecycle -------------------------------------------------------- #
    def open(self) -> None:
        # Set before the @spaces.GPU call: the worker compares against it to
        # recognise its own traffic.
        with _LEASE_LOCK:
            LEASE_IDS[self.slot] += 1
            self.lease_id = LEASE_IDS[self.slot]
        self.reference_is_default = True   # a fresh worker starts on the default
        self._alive.set()
        self.last_error = None
        if self._reader is None or not self._reader.is_alive():
            self._reader = threading.Thread(target=self._drain,
                                            name=f"gpu-rpc-reader-{self.slot}",
                                            daemon=True)
            self._reader.start()

    def close(self, reason: str = "lease ended") -> None:
        self._alive.clear()
        self.last_error = reason
        # No stop sentinel on the request queue: close() runs after the worker
        # is gone, so the sentinel would sit there and shut down the NEXT
        # lease's lanes instead. Lanes end on their own deadline.
        with self._cv:
            for slot in self._pending.values():
                slot[:] = [_LEASE_GONE, reason]
            self._cv.notify_all()

    @property
    def live(self) -> bool:
        return self._alive.is_set()

    # -- plumbing --------------------------------------------------------- #
    def _drain(self) -> None:
        while True:
            try:
                seq, ok, payload = self.res_q.get(timeout=1.0)
            except _queue.Empty:
                if not self._alive.is_set():
                    return
                continue
            except (EOFError, OSError):
                return
            if seq is None:                     # worker-side stop echo
                continue
            with self._cv:
                slot = self._pending.get(seq)
                if slot is not None:
                    slot[:] = [ok, payload]
                self._cv.notify_all()

    def call(self, method: str, *args, lane: str = LANE_ENGINE,
             timeout: float = CALL_TIMEOUT):
        if not self._alive.is_set():
            raise LeaseError(self.last_error or "no GPU lease held")
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        slot: list = []
        with self._cv:
            self._pending[seq] = slot
        try:
            self.req_q.put((self.lease_id, seq, lane, method, args))
            end = time.monotonic() + timeout
            with self._cv:
                while not slot:
                    left = end - time.monotonic()
                    if left <= 0:
                        raise LeaseError(f"GPU call {method!r} timed out after {timeout:.0f}s")
                    self._cv.wait(left)
                ok, payload = slot[0], slot[1]
        finally:
            with self._cv:
                self._pending.pop(seq, None)
        # `==`, not `is`: the sentinel crosses a pickle boundary, so identity
        # never matches.
        if ok == _LEASE_GONE:       # worker is gone, or refused a stale lease
            raise LeaseError(str(payload))
        if ok is not True:          # the worker ran the call and it raised
            raise RuntimeError(str(payload))
        return payload

    # -- the surface web.py uses ------------------------------------------ #
    def push_frame(self, jpeg: bytes) -> dict:
        return self.call("push_frame", jpeg, lane=LANE_CROP, timeout=30.0)

    def prime(self) -> dict:
        return self.call("prime", timeout=90.0)

    def step(self, avatar_f32: np.ndarray, user_f32: np.ndarray) -> dict:
        return self.call("step", avatar_f32, user_f32, timeout=30.0)

    def reprime(self) -> dict:
        return self.call("reprime", timeout=60.0)

    def set_reference(self, image: bytes | None) -> float:
        res = self.call("set_reference", image, timeout=60.0)
        self.reference_is_default = bool(res.get("is_default", True))
        return float(res.get("secs", 0.0))

    def reset_cropper(self) -> None:
        self.call("reset_cropper", lane=LANE_CROP, timeout=30.0)

    # -- speech ------------------------------------------------------------ #
    # Crop lane, not engine lane: a synthesis queued behind a step would stall
    # video for a block.
    def tts_set_voice(self, ref_bytes, ref_text, key="") -> dict:
        return self.call("tts_set_voice", ref_bytes, ref_text, key,
                         lane=LANE_CROP, timeout=120.0)

    def tts_synth(self, text: str) -> dict:
        return self.call("tts_synth", text, lane=LANE_CROP, timeout=60.0)

    def stt_transcribe(self, pcm: bytes, sr: int = 16000) -> dict:
        return self.call("stt_transcribe", pcm, sr, lane=LANE_CROP, timeout=60.0)

    def stats(self) -> dict:
        try:
            return self.call("stats", timeout=10.0)
        except Exception:
            return {}


# =========================================================================== #
# session registry
# =========================================================================== #
# A conversation spans two connections -- the /run_session job holding the
# lease and the /ws streaming it. run_session mints a token, hands it to the
# client in `ready`, and the client presents it when opening /ws.
_SESSIONS: dict[str, "GPUProxy"] = {}
_SESSIONS_LOCK = threading.Lock()


def new_session() -> tuple[str, "GPUProxy"] | None:
    """Claim a slot and register a proxy for it. None when at capacity."""
    slot = claim_slot()
    if slot is None:
        return None
    token = secrets.token_urlsafe(16)
    proxy = GPUProxy(slot)
    with _SESSIONS_LOCK:
        _SESSIONS[token] = proxy
    return token, proxy


def get_session(token: str | None) -> "GPUProxy | None":
    if not token:
        return None
    with _SESSIONS_LOCK:
        return _SESSIONS.get(token)


def end_session(token: str) -> None:
    with _SESSIONS_LOCK:
        proxy = _SESSIONS.pop(token, None)
    if proxy is not None:
        release_slot(proxy.slot)

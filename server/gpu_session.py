"""In-process GPU backend for a dedicated GPU (local box, own cloud, or a
paid Space).

Same module surface as the ZeroGPU variant (``streaming-zerogpu`` branch), so
``server/web.py`` and ``server/conversation.py`` are identical across the two:
sessions claim a slot, get a proxy, and talk to a ``_Worker`` that owns the
engine, the face cropper, the frame ring and JPEG encoding. Here the worker
lives in this process and calls are direct — no lease, no fork, no queues —
and a session runs until the client leaves (or ``AVATAR_SESSION_SECONDS``, if
set).
"""

from __future__ import annotations

import io
import logging
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

log = logging.getLogger("avatar.gpu")

# The engine's KV cache and rollout are per-conversation state, and there is
# one engine, so one conversation at a time. Raising this requires one engine
# (and its VRAM) per slot.
MAX_SESSIONS = int(os.environ.get("AVATAR_MAX_SESSIONS", "1"))
# 0 = unlimited: the session ends when the client leaves.
SESSION_SECONDS = int(os.environ.get("AVATAR_SESSION_SECONDS", "0"))
LEASE_SECONDS = SESSION_SECONDS          # /healthz compatibility
GPU_SIZE = "dedicated"

BLOCK_FRAMES = 10
OUT_JPEG_QUALITY = int(os.environ.get("AVATAR_JPEG_QUALITY", "72"))
OUT_JPEG_SUBSAMPLING = int(os.environ.get("AVATAR_JPEG_SUBSAMPLING", "2"))
REPRIME_XFADE = int(os.environ.get("AVATAR_REPRIME_XFADE", "10"))

_SLOT_LOCK = threading.Lock()
_SLOTS_FREE: list[int] = list(range(MAX_SESSIONS))
LEASE_IDS: list[int] = [0] * MAX_SESSIONS
_LEASE_LOCK = threading.Lock()

# slot -> (lease_id, _Worker); written by gpu_worker_body, read by the proxy.
_WORKERS: dict[int, tuple[int, "_Worker"]] = {}
_WORKERS_LOCK = threading.Lock()


def claim_slot() -> int | None:
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
    """One session's lifetime. Yields ``ready``, heartbeats, then ``expired``.

    The factories return already-loaded singletons (app.py builds everything at
    boot), so ``ready`` is immediate.
    """
    t0 = time.monotonic()
    worker = _Worker(engine_factory(),
                     cropper_factory() if cropper_factory is not None else None,
                     speech_factory() if speech_factory is not None else None)
    with _WORKERS_LOCK:
        _WORKERS[slot] = (lease_id, worker)
    deadline = t0 + SESSION_SECONDS if SESSION_SECONDS else None
    try:
        yield {"event": "ready", "warm_seconds": 0.0,
               "session_seconds": float(SESSION_SECONDS)}
        while True:
            time.sleep(1.0)
            with _WORKERS_LOCK:
                current = _WORKERS.get(slot)
            if current is None or current[0] != lease_id:
                return
            if deadline is None:
                yield {"event": "alive"}         # ignored by the client
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            yield {"event": "tick", "remaining": round(remaining, 1)}
    finally:
        with _WORKERS_LOCK:
            if _WORKERS.get(slot, (None,))[0] == lease_id:
                _WORKERS.pop(slot, None)
        log.info("session slot %d closed after %.0fs", slot, time.monotonic() - t0)
    yield {"event": "expired"}


class LeaseError(RuntimeError):
    """No GPU session is currently held, or it went away mid-call."""


class GPUProxy:
    """Direct in-process handle on this session's worker."""

    def __init__(self, slot: int = 0) -> None:
        self.slot = slot
        self.lease_id = 0
        self._alive = threading.Event()
        self.reference_is_default = True
        self.last_error: str | None = None

    # -- lifecycle -------------------------------------------------------- #
    def open(self) -> None:
        with _LEASE_LOCK:
            LEASE_IDS[self.slot] += 1
            self.lease_id = LEASE_IDS[self.slot]
        self.reference_is_default = True
        self._alive.set()
        self.last_error = None

    def close(self, reason: str = "session ended") -> None:
        self._alive.clear()
        self.last_error = reason
        with _WORKERS_LOCK:
            if _WORKERS.get(self.slot, (None,))[0] == self.lease_id:
                _WORKERS.pop(self.slot, None)

    @property
    def live(self) -> bool:
        return self._alive.is_set()

    def _worker(self) -> "_Worker":
        if not self._alive.is_set():
            raise LeaseError(self.last_error or "no GPU session held")
        with _WORKERS_LOCK:
            entry = _WORKERS.get(self.slot)
        if entry is None or entry[0] != self.lease_id:
            raise LeaseError("that GPU session has ended")
        return entry[1]

    # -- the surface web.py / conversation.py use -------------------------- #
    def push_frame(self, jpeg: bytes) -> dict:
        return self._worker().push_frame(jpeg)

    def prime(self) -> dict:
        return self._worker().prime()

    def step(self, avatar_f32: np.ndarray, user_f32: np.ndarray) -> dict:
        return self._worker().step(avatar_f32, user_f32)

    def reprime(self) -> dict:
        return self._worker().reprime()

    def set_reference(self, image: bytes | None) -> float:
        res = self._worker().set_reference(image)
        self.reference_is_default = bool(res.get("is_default", True))
        return float(res.get("secs", 0.0))

    def reset_cropper(self) -> None:
        self._worker().reset_cropper()

    def tts_set_voice(self, ref_bytes, ref_text, key="") -> dict:
        return self._worker().tts_set_voice(ref_bytes, ref_text, key)

    def tts_synth(self, text: str) -> dict:
        return self._worker().tts_synth(text)

    def stt_transcribe(self, pcm: bytes, sr: int = 16000) -> dict:
        return self._worker().stt_transcribe(pcm, sr)

    def stats(self) -> dict:
        try:
            return self._worker().stats()
        except Exception:
            return {}


# =========================================================================== #
# session registry
# =========================================================================== #
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

"""ZeroGPU held-lease GPU worker + parent-side RPC proxy.

Why this module exists
----------------------
The dedicated-GPU Space keeps one CUDA context alive for the process lifetime
and calls ``engine.step()`` every 400 ms, rolling a KV cache forward. Neither
half of that is available on ZeroGPU:

* there is **no GPU outside** ``@spaces.GPU``, so ``engine.load()``'s warm-up and
  ``FaceCropper``'s SFD detections cannot run in the web process; and
* every ``@spaces.GPU`` call **forks a fresh worker**, so a per-block decorator
  would throw the KV cache away between blocks (module globals set in one call
  are simply absent in the next).

So the GPU is leased **once per conversation**. :func:`run_session` is a
``@spaces.GPU`` generator: entering it forks a worker that restores the packed
weights, warms the engine, announces ``__READY__`` and then serves an RPC loop
until the lease expires. The web process talks to it through a pair of
fork-context queues created at *import* time — created in the parent before any
fork, they are inherited by the worker with their pipe fds intact, so the parent
can keep pushing work into a worker that is already blocked on ``get()``.

Wire economy
------------
The naive split (parent crops, ships ``[10,512,512,3]`` in and out) would push
~16 MB per 400 ms block through a pickle pipe. Instead the worker owns the face
cropper, the cropped-frame ring *and* JPEG encoding, so the traffic is ~25 KB of
webcam JPEG in and ~200 KB of encoded frames out — about 1.5% of that.

Threading
---------
The worker runs three interchangeable dispatch lanes so that the three kinds of
work in flight — a 400 ms ``step()``, a ~20 ms crop, and a ~740 ms speech
synthesis — never head-of-line block each other in the queue. They still
serialise on the GPU; the lanes only stop a long job from delaying the dispatch
of a short one. Threads only — a ``@spaces.GPU`` fork is daemonic and cannot
spawn child processes.

Speech
------
The worker also hosts OmniVoice TTS and Whisper STT (``server/speech.py``),
which replaced ElevenLabs. That moves speech from a network call in the web
process onto the same leased GPU as the avatar engine, so the two now share a
budget — see ``server/speech.py`` for the arithmetic.
"""

from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing as _mp
import os
import queue as _queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

log = logging.getLogger("avatar.gpu")

# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
# Seconds of *conversation* the visitor gets. The clock starts when the engine
# is warm and __READY__ goes out, not when the GPU is granted -- otherwise the
# warm-up silently eats a chunk of the session (measured: 11 s cold / 4 s warm,
# i.e. a "90 s" lease delivering 76 s of talking).
SESSION_SECONDS = int(os.environ.get("AVATAR_SESSION_SECONDS", "90"))
# Headroom reserved for that warm-up when asking ZeroGPU for the lease. Covers
# the engine warm (4-13 s) plus loading the live ASR model, which has to happen
# in here rather than at import -- see server/speech.py.
WARM_ALLOWANCE = float(os.environ.get("AVATAR_WARM_ALLOWANCE", "25.0"))
# Stop serving this many seconds before the lease actually lapses, so the worker
# tears down cleanly instead of being killed mid-step.
LEASE_MARGIN = float(os.environ.get("AVATAR_LEASE_MARGIN", "3.0"))
# What @spaces.GPU(duration=...) actually requests. Larger values queue lower
# and are checked against the visitor's remaining quota as a whole, so this is
# the conversation plus just enough slack to cover a cold warm-up.
LEASE_SECONDS = int(SESSION_SECONDS + WARM_ALLOWANCE + LEASE_MARGIN)
# large = half the Blackwell card (48 GB, 1x quota); xlarge = the full card
# (96 GB, 2x quota) and therefore full SM count.
#
# Measured on the half-MIG with the real pipeline: engine step 202-224 ms and
# 213-236 ms including the fork round-trip, against a 400 ms block budget, with
# 0 late blocks and 0 dropped frames over 187 blocks. xlarge would cost 2x the
# visitor's quota to buy headroom that is already there, so: large.
GPU_SIZE = os.environ.get("AVATAR_GPU_SIZE", "large")
# How long a parent-side RPC waits before giving up on the worker.
CALL_TIMEOUT = float(os.environ.get("AVATAR_CALL_TIMEOUT", "120.0"))

BLOCK_FRAMES = 10
OUT_JPEG_QUALITY = int(os.environ.get("AVATAR_JPEG_QUALITY", "72"))
OUT_JPEG_SUBSAMPLING = int(os.environ.get("AVATAR_JPEG_SUBSAMPLING", "2"))
REPRIME_XFADE = int(os.environ.get("AVATAR_REPRIME_XFADE", "10"))

# --------------------------------------------------------------------------- #
# fork-context queues -- MUST be constructed at import, in the parent, before
# any @spaces.GPU call forks. A queue made inside the worker would not be
# visible to the parent, and one made lazily after the first fork would not be
# inherited by it.
# --------------------------------------------------------------------------- #
_CTX = _mp.get_context("fork")
REQ_Q: Any = _CTX.Queue()      # parent -> worker: (seq, lane, method, args)
RES_Q: Any = _CTX.Queue()      # worker -> parent: (seq, ok, payload)

_STOP = "__stop__"
_LEASE_GONE = "__lease_gone__"     # distinguishes "worker vanished" from "call raised"
LANE_ENGINE = "engine"
LANE_CROP = "crop"

# Which lease the parent is talking to. Bumped by GPUProxy.open() and passed to
# the worker as a CALL ARGUMENT -- deliberately not left to fork inheritance.
#
# ZeroGPU reuses worker processes ("engine warm in 0.0s" on a second lease gives
# it away), and a reused worker still holds the LEASE_ID from whenever it was
# first forked. Relying on inheritance therefore made the worker reject every
# request from the new lease, and the session hung with 0 frames cropped.
#
# The id exists because a session ending on lease expiry can leave a `step` in
# flight; without it the next lease runs that stale call against an engine that
# was never primed ("call start_session() before step()").
LEASE_ID = 0


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
        discarded, exactly as the dedicated Space does — the client shows the
        reference photo until the first generated block lands."""
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
    # Phrase-at-a-time rather than one call per utterance: OmniVoice cannot
    # stream below chunk granularity, so the phrase IS the streaming unit, and
    # a short phrase also keeps each GPU burst small enough for the avatar
    # engine's per-block budget.
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
           my_lease: int) -> None:
    """Pull work off ``REQ_Q`` and answer on ``RES_Q`` until stopped or expired.

    Lanes are interchangeable pullers, not routed queues: two of them run so a
    ~20 ms ``push_frame`` never waits behind a ~400 ms ``step``, which is the
    same overlap the dedicated Space got from its separate crop/GPU executors.
    ``web.py`` keeps at most one crop and one step in flight, and speech adds a
    third, so three is enough. They still serialise on the GPU -- the point is
    that a ~740 ms synthesis does not head-of-line block a 20 ms crop.
    """
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            item = REQ_Q.get(timeout=min(0.25, remaining))
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
            log.info("dropping stale request %s from lease %s (serving %s)",
                     method, lease_id, my_lease)
            RES_Q.put((seq, _LEASE_GONE, "that GPU session has ended"))
            continue
        try:
            payload = getattr(worker, method)(*args)
            RES_Q.put((seq, True, payload))
        except Exception as exc:                # never kill the lane on one bad call
            log.exception("rpc %s failed", method)
            RES_Q.put((seq, False, f"{type(exc).__name__}: {exc}"))


def _pin_cudnn_benchmark_off() -> None:
    """Make ``cudnn.benchmark = True`` a no-op for the life of the worker.

    ``face_alignment``'s SFD detector sets it on *every* ``detect_from_image``,
    so each new input shape pays a full cuDNN autotune. ``detect_box`` rescales
    to a fixed HEIGHT, so the width still tracks the source aspect ratio: the
    default reference and the square webcam frames share one shape, but an
    arbitrary user upload is a shape nothing has warmed.

    On the dedicated Space that cost was paid once per process lifetime (which
    is why set_reference is documented there at 56 ms). On ZeroGPU every lease
    is a fresh fork, so it was paid *every session* -- measured at 3.2 s for
    set_reference, 3.8 s to prime and a 5.7 s first step, which left the block
    loop 45 blocks behind and the avatar frozen on the reference photo.

    The engine already asks for ``benchmark = False`` (see AvatarEngine.load);
    this just stops face_alignment from overriding it. Heuristic algo choice
    costs a little per conv and saves seconds of autotune.
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
                    speech_factory=None, lease_id: int = 0):
    """The body of the lease. A generator so the caller can stream progress.

    Yields ``"__READY__"`` once the engine is warm, then a ``{"remaining": n}``
    heartbeat roughly once a second, then ``"__EXPIRED__"``.
    """
    t_lease_start = time.monotonic()
    # Hard stop: whatever else happens, be done before ZeroGPU reclaims us.
    hard_deadline = t_lease_start + lease_seconds - LEASE_MARGIN

    t0 = time.perf_counter()
    _pin_cudnn_benchmark_off()
    engine = engine_factory()
    # Weights were built and packed in the parent; this restores them to VRAM
    # and pays the reference-latent + cuDNN warm-up that needs a real GPU.
    engine.warm()
    cropper = cropper_factory() if cropper_factory is not None else None
    if cropper is not None:
        # index.html always sends a square frame, which the cropper rescales to
        # 360x360 -- a shape the engine's own 360x480 warm-up does not cover, and
        # whose first cuDNN autotune otherwise lands on the first webcam frame.
        rng = np.random.default_rng(0)
        size = int(os.environ.get("AVATAR_CAM_WARM_SIZE", "384"))
        try:
            cropper.crop(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
            if hasattr(cropper, "reset"):
                cropper.reset()
        except Exception:
            log.exception("cropper warm-up failed (non-fatal)")
    warm_s = time.perf_counter() - t0
    # The session clock starts HERE, so a cold warm-up costs the visitor slack
    # from WARM_ALLOWANCE rather than conversation time. Capped by the hard
    # deadline in case the warm-up overran that allowance.
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

    # The session clock starts HERE -- after EVERYTHING that has to happen
    # before the first block, speech included. Setting it before speech init
    # handed the visitor 63 s of a 90 s session, because loading the AoTI
    # package and the ASR model ran on their clock.
    deadline = min(time.monotonic() + SESSION_SECONDS, hard_deadline)
    log.info("lease up: warm %.1fs total, %.0fs of session",
             time.monotonic() - t_lease_start, deadline - time.monotonic())

    worker = _Worker(engine, cropper, speech)

    # Drain anything a previous, aborted session left behind, so this lease does
    # not answer a stale request with a fresh sequence number.
    while True:
        try:
            REQ_Q.get_nowait()
        except _queue.Empty:
            break

    stop = threading.Event()
    lanes = [threading.Thread(target=_serve,
                              args=(worker, deadline, stop, lease_id),
                              name=f"gpu-lane-{i}", daemon=True)
             for i in range(3)]
    for t in lanes:
        t.start()

    yield {"event": "ready", "warm_seconds": round(warm_s, 1),
           "session_seconds": round(deadline - time.monotonic(), 1)}

    while any(t.is_alive() for t in lanes):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))
        yield {"event": "tick", "remaining": round(max(0.0, deadline - time.monotonic()), 1)}

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

    def __init__(self) -> None:
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
        global LEASE_ID
        # Must happen before the @spaces.GPU call forks: the worker reads this
        # global to recognise its own traffic.
        LEASE_ID += 1
        self.lease_id = LEASE_ID
        self.reference_is_default = True   # a fresh worker starts on the default
        self._alive.set()
        self.last_error = None
        if self._reader is None or not self._reader.is_alive():
            self._reader = threading.Thread(target=self._drain, name="gpu-rpc-reader",
                                            daemon=True)
            self._reader.start()

    def close(self, reason: str = "lease ended") -> None:
        self._alive.clear()
        self.last_error = reason
        # NOTE: deliberately does NOT put a stop sentinel on REQ_Q. close() runs
        # in run_session's finally, i.e. *after* the worker is already gone, so
        # the sentinel would just sit in the queue -- and the NEXT lease's lanes
        # would read it and shut down instantly ("lease over after 5s", right
        # after warm-up). The lanes end on their own deadline; the client
        # cancelling the /run_session job ends the lease early.
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
                seq, ok, payload = RES_Q.get(timeout=1.0)
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
            REQ_Q.put((self.lease_id, seq, lane, method, args))
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
        # `==`, not `is`: the sentinel crosses a pickle boundary coming back
        # from the worker, so the parent unpickles a DIFFERENT string object.
        # With `is` this test never fired, the truthy sentinel slipped past the
        # `not ok` check, and a rejected call returned its error message as if
        # it were a result -- which then hung the session instead of failing it.
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
    # These run on the crop lane, not the engine lane: the engine lane is the
    # 400 ms block loop, and queueing a ~740 ms synthesis behind (or in front
    # of) a step would stall video for a whole block. The two lanes are
    # interchangeable pullers, so this just means "not behind the steps".
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


# One proxy per process; the queues are module-global singletons, so there can
# only ever be one lease in flight anyway.
PROXY = GPUProxy()

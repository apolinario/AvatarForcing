"""Whisper transcription in a dedicated child process.

The web process must never run torch inference: ZeroGPU patches
``torch.cuda.is_available()`` to True there, so transformers takes a CUDA code
path during generation and initialises a real CUDA context — after which every
``@spaces.GPU`` fork dies in ``worker_init`` with "No CUDA GPUs are
available" until a factory reboot. ``device="cpu"`` does not prevent it.

So this runs in a child process, where a CUDA context is harmless. The child
is forked early (before the big models are built) so it inherits a small
parent, loads its model once, and serves requests.
"""

from __future__ import annotations

import io
import logging
import multiprocessing as _mp
import os
import threading

log = logging.getLogger("avatar.cpu_asr")

# Seconds of the uploaded clip actually used. MUST match server/speech.py's
# REF_MAX_SECS: the clone conditions on audio AND its transcript, so
# transcribing a different span than the clone hears would mis-condition it.
REF_MAX_SECS = float(os.environ.get("AVATAR_REF_MAX_SECS", "15"))

_CTX = _mp.get_context("fork")


def _worker(model_name: str, req, res) -> None:
    """Child process: load the model once, then transcribe on demand."""
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import pipeline

        pipe = pipeline("automatic-speech-recognition", model=model_name,
                        device="cpu", torch_dtype=torch.float32)
        res.put((True, "__ready__"))
    except Exception as exc:            # report instead of dying silently
        res.put((False, f"load failed: {type(exc).__name__}: {exc}"))
        return

    while True:
        item = req.get()
        if item is None:
            return
        try:
            wav, sr = sf.read(io.BytesIO(item), dtype="float32", always_2d=False)
            wav = np.asarray(wav)
            if wav.ndim > 1:                    # stereo -> mono
                wav = wav.mean(axis=1)
            if wav.shape[0] > REF_MAX_SECS * sr:
                wav = wav[: int(REF_MAX_SECS * sr)]
            out = pipe({"array": wav, "sampling_rate": int(sr)})
            res.put((True, (out or {}).get("text", "").strip()))
        except Exception as exc:
            res.put((False, f"{type(exc).__name__}: {exc}"))


class CPUTranscriber:
    """Parent-side handle. Serialised: one clip at a time is plenty."""

    def __init__(self, model_name: str) -> None:
        self._req = _CTX.Queue()
        self._res = _CTX.Queue()
        self._lock = threading.Lock()
        self.ready = False
        self._proc = _CTX.Process(target=_worker,
                                  args=(model_name, self._req, self._res),
                                  name="cpu-asr", daemon=True)
        self._proc.start()
        try:
            ok, msg = self._res.get(timeout=300)
            self.ready = bool(ok)
            if not ok:
                log.error("CPU ASR child failed to start: %s", msg)
            else:
                log.info("CPU ASR child ready (%s, pid %s)", model_name,
                         self._proc.pid)
        except Exception:
            log.exception("CPU ASR child did not report readiness")

    def transcribe(self, audio_bytes: bytes, timeout: float = 300.0) -> str:
        if not self.ready or not self._proc.is_alive():
            raise RuntimeError("CPU transcriber is not running")
        with self._lock:
            self._req.put(audio_bytes)
            ok, payload = self._res.get(timeout=timeout)
        if not ok:
            raise RuntimeError(str(payload))
        return str(payload)

"""Whisper transcription in a dedicated child process.

Why not just call the pipeline in the web process
-------------------------------------------------
Because it bricks every subsequent GPU lease. ZeroGPU patches
``torch.cuda.is_available()`` to return True in the web process, so transformers
takes a CUDA code path during generation and initialises a real CUDA context —
after which every ``@spaces.GPU`` fork dies in ``worker_init``:

    torch.init(nvidia_uuid) -> torch.Tensor([0]).cuda()
    RuntimeError: No CUDA GPUs are available

``device="cpu"`` does NOT prevent this. It was reproducible: upload a voice
clip, and the next lease — and every lease after it — failed until a factory
reboot. Sessions that never touched ``/voice`` were fine.

So the rule is: **the web process must never run torch inference.** This runs it
in a child instead, where a CUDA context (if one is created at all) is harmless
because it belongs to a different process.

The child is forked early, before the big models are built, so it inherits a
small parent. It loads its own model once and then serves requests, which keeps
per-request latency to the transcription itself.
"""

from __future__ import annotations

import io
import logging
import multiprocessing as _mp
import threading

log = logging.getLogger("avatar.cpu_asr")

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
            if wav.shape[0] > 30 * sr:          # more than this is pointless
                wav = wav[: 30 * sr]
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

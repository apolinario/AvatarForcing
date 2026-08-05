"""Semantic end-of-turn detection (pipecat-ai/smart-turn-v3).

Energy VAD cannot tell "I'm done" from "I'm thinking"; a fixed silence timeout
must cover the worst case, and every turn pays it before any compute starts.
Smart Turn — a Whisper-tiny encoder plus a linear head, 8M params — answers
"has this person finished?" from the waveform, so the VAD becomes a short
trigger and this model makes the real call.

It runs in the WEB process, on the CPU, deliberately:

* it is ~10-30 ms there and fires every ~100 ms while someone pauses;
  round-tripping to the GPU worker would cost more than the inference;
* it needs no GPU lease, so turn detection works outside a session; and
* onnxruntime is not torch, so it cannot initialise the CUDA context that
  poisons ``@spaces.GPU`` forks (see ``server/cpu_asr.py``).

Preprocessing
-------------
``WhisperFeatureExtractor(chunk_length=8)``, 16 kHz, ``do_normalize=True``
(without it the model returns ~0.98 for everything, silence included), the
LAST 8 s of the utterance, padded on the **LEFT** so the utterance ends at the
window's right-hand edge — the moment being asked about. Left-padding scores
96.8% vs 90.4% for the upstream reference's right-padding on the model's own
labelled test set (``smart-turn-data-v3.1-test``); a held-out shard driving
this exact module scores 93.9% at ~50 ms per probe.

The ONNX graph already ends in a sigmoid: the output is P(the speaker has
finished) despite the tensor being named ``logits``.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

log = logging.getLogger("avatar.turn")

REPO = os.environ.get("SMART_TURN_REPO", "pipecat-ai/smart-turn-v3")
# The CPU export on purpose -- see the module docstring.
FILE = os.environ.get("SMART_TURN_FILE", "smart-turn-v3.2-cpu.onnx")
SR = 16000
WINDOW_SECS = 8
WINDOW_SAMPLES = SR * WINDOW_SECS


class SmartTurn:
    """Thread-safe wrapper. Construct freely; the model loads on first use."""

    def __init__(self) -> None:
        self._sess = None
        self._fe = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def available(self) -> bool:
        return not self._failed

    def load(self) -> bool:
        """Build the session. Returns False if the model is unusable."""
        if self._sess is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._sess is not None:
                return True
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from transformers import WhisperFeatureExtractor

                path = hf_hub_download(REPO, FILE)
                opts = ort.SessionOptions()
                # One turn probe at a time, and the caller is already off the
                # event loop -- extra intra-op threads would just contend with
                # the session's own work.
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self._sess = ort.InferenceSession(
                    path, sess_options=opts, providers=["CPUExecutionProvider"])
                self._fe = WhisperFeatureExtractor(chunk_length=WINDOW_SECS)
                log.info("smart-turn ready (%s)", FILE)
                return True
            except Exception:
                # Never fatal: without it the brain falls back to a plain
                # hangover, which is the old behaviour.
                log.exception("smart-turn unavailable; falling back to VAD hangover")
                self._failed = True
                return False

    def probability(self, audio_f32: np.ndarray) -> float | None:
        """P(the speaker has finished), or None if the model is unusable.

        ``audio_f32`` is mono 16 kHz in [-1, 1]; only its last 8 s are used.
        """
        if not self.load():
            return None
        x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return None
        x = x[-WINDOW_SAMPLES:]
        if x.size < WINDOW_SAMPLES:
            # LEFT pad: the utterance must end at the window's right edge, which
            # is the moment being asked about. See the module docstring for the
            # measurement -- this is worth 6 accuracy points over padding right.
            x = np.concatenate([np.zeros(WINDOW_SAMPLES - x.size, dtype=np.float32), x])
        try:
            feats = self._fe(x, sampling_rate=SR, return_tensors="np",
                             padding="max_length", max_length=WINDOW_SAMPLES,
                             truncation=True, do_normalize=True)["input_features"]
            feats = np.asarray(feats, dtype=np.float32)
            if feats.ndim == 2:                      # [80, 800] -> [1, 80, 800]
                feats = feats[None]
            out = self._sess.run(None, {"input_features": feats})
            p = float(np.asarray(out[0]).reshape(-1)[0])
        except Exception:
            log.exception("smart-turn inference failed")
            return None
        if not np.isfinite(p):
            return None
        # The graph ends in a sigmoid, so this is already a probability. Guard
        # anyway: a future export that returned a raw logit would otherwise be
        # read as "complete" for every value above 0, silently.
        if p < 0.0 or p > 1.0:
            p = 1.0 / (1.0 + np.exp(-p))
        return p


# One per process. The session is stateless, so sharing it across concurrent
# conversations is safe and saves loading the model N times.
DETECTOR = SmartTurn()

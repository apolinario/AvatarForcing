"""OmniVoice TTS + Whisper STT, running inside the GPU worker.

OmniVoice is a masked diffusion LM: it predicts the output duration up front
and denoises the whole token sequence over ``num_step`` parallel passes, so no
audio exists until the final step and there is nothing to stream below chunk
granularity. Synthesis is therefore phrase-at-a-time — the same granularity
the conversation brain produces as LLM tokens arrive.

Budget: RTF ~0.23 at ``num_step=32`` (a 3 s phrase costs ~740 ms of GPU),
sharing the card with an avatar engine that uses ~220 ms of every 400 ms
block. Short phrases keep each burst inside what the block loop absorbs.
"""

from __future__ import annotations

import io
import logging
import os
import time

import numpy as np
import torch   # module scope is safe: this module is only imported in the worker

log = logging.getLogger("avatar.speech")

# Same settings as the official k2-fsa/OmniVoice demo.
NUM_STEP = 32
GUIDANCE_SCALE = 2.0

# AoT-Inductor package for OmniVoice.forward, built by
# multimodalart/omnivoice-aoti-compile. Empty string disables it.
AOTI_REPO = os.environ.get("OMNIVOICE_AOTI_REPO", "multimodalart/omnivoice-aoti")

TTS_SR = 24000          # OmniVoice output
WIRE_SR = 16000         # the avatar protocol / engine sample rate

# Target seconds of audio per synthesis call. The real split is punctuation
# driven (chunk_text_punctuation will not break below min_chunk_len), so this is
# an upper bound, not a guarantee -- a single long clause still generates whole.
PHRASE_TARGET_SECS = 1.5

# Seconds of the uploaded reference clip used for the clone. Must match
# server/cpu_asr.py's REF_MAX_SECS: the transcript is produced from the first
# REF_MAX_SECS, so cloning from a longer span would pair audio with text that
# does not describe it. (OmniVoice trims at 20 s of its own accord; this is
# tighter and, crucially, agrees with the transcriber.)
REF_MAX_SECS = float(os.environ.get("AVATAR_REF_MAX_SECS", "15"))

# Silence padding is disabled so consecutive phrases butt together without a
# gap; a short fade still runs to keep the joins from clicking.
PAD_DURATION = 0.0
FADE_DURATION = 0.02


def _to_wire_pcm(audio_f32: np.ndarray) -> bytes:
    """OmniVoice 24 kHz float32 -> protocol 16 kHz int16 little-endian."""
    import soxr

    a = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return b""
    a = soxr.resample(a, TTS_SR, WIRE_SR, quality="VHQ")
    return np.clip(a * 32767.0, -32768, 32767).astype("<i2").tobytes()


class PaddedAOTI:
    """Pad the sequence into the compiled graph's shape class, then slice back.

    The export specialised on ``seq % 8 == 6`` (the model pads its sequence to
    a multiple of the 8 codebooks, so the modulo is baked into the graph), but
    real phrases land on any residue. Rather than compile one artifact per
    residue, pad up to the next conforming length (at most 7 positions), run,
    and slice the answer back. The padded
    positions are masked out of attention in BOTH directions, so no real
    position can attend to them and the real logits are unchanged.
    """

    RESIDUE = 6
    MULT = 8

    def __init__(self, inner, pad_token_id: int):
        self.inner = inner
        self.pad_token_id = pad_token_id
        self.n_padded = 0

    def __call__(self, *args, **kwargs):
        ids = kwargs.get("input_ids")
        if ids is None:
            return self.inner(*args, **kwargs)
        S = ids.shape[-1]
        pad = (self.RESIDUE - S) % self.MULT
        if pad == 0:
            return self.inner(*args, **kwargs)
        self.n_padded += 1
        S2 = S + pad
        kw = dict(kwargs)
        kw["input_ids"] = torch.nn.functional.pad(ids, (0, pad),
                                                  value=self.pad_token_id)
        am = kwargs.get("audio_mask")
        if am is not None:                       # padding is not audio
            kw["audio_mask"] = torch.nn.functional.pad(am, (0, pad), value=False)
        attn = kwargs.get("attention_mask")
        if attn is not None:
            # [B,1,S,S] -> [B,1,S2,S2], False everywhere the padding touches, so
            # real queries cannot see pad keys and pad queries see nothing.
            new = attn.new_zeros((*attn.shape[:-2], S2, S2))
            new[..., :S, :S] = attn
            # Let each pad row attend to itself. SDPA reads True as "take part",
            # so an all-False row is fully masked and softmax over all -inf is
            # NaN. Those rows are sliced off, but a self-only row costs nothing
            # and keeps NaN out of the tensor entirely.
            idx = torch.arange(S, S2, device=attn.device)
            new[..., idx, idx] = True
            kw["attention_mask"] = new
        out = self.inner(*args, **kw)
        logits = out.logits if hasattr(out, "logits") else out
        logits = logits[..., :S, :]              # drop the padded positions
        if hasattr(out, "logits"):
            out.logits = logits
            return out
        return logits


def aoti_loader(module, package_dir):
    """Supply weights INCLUDING non-persistent buffers, under both spellings.

    ``state_dict()`` omits non-persistent buffers (the rotary cache is one);
    ``named_buffers()`` has them. The artifact refers to constants both by
    dotted and underscore-flattened names, so both spellings go in and the
    extra keys are ignored.
    """
    from pathlib import Path

    from spaces.zero.torch.aoti import LazyAOTIModel, PACKAGE_FILENAME

    archive = Path(package_dir) / "root" / PACKAGE_FILENAME
    model = LazyAOTIModel(str(archive))
    weights = {}
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        weights[name] = tensor
        weights[name.replace(".", "_")] = tensor
    pad_id = getattr(module.config, "pad_token_id", 0) or 0
    module.forward = PaddedAOTI(model.with_weights(weights), pad_id)


class Speech:
    """Owns the OmniVoice model and its Whisper pipe. Lives in the GPU worker."""

    def __init__(self, model, asr_model=None, asr_processor=None) -> None:
        self.model = model
        # AoTI is bound at import in app.py (CPU-only work that must not run on
        # the visitor's lease); whether it took is whether forward is wrapped.
        self.aoti = isinstance(getattr(model, "forward", None), PaddedAOTI)
        self._prompt = None          # cached VoiceClonePrompt for this session
        self._voice_key: str | None = None

        # ASR weights were loaded and packed at import; only the pipeline
        # object is assembled here. Building a transformers pipeline in the web
        # process would initialise CUDA there and poison every later fork;
        # moving weights with .to("cuda") is fine (ZeroGPU intercepts it).
        if asr_model is not None and asr_processor is not None:
            t0 = time.perf_counter()
            try:
                from transformers import pipeline

                self.model._asr_pipe = pipeline(
                    "automatic-speech-recognition",
                    model=asr_model,
                    tokenizer=asr_processor.tokenizer,
                    feature_extractor=asr_processor.feature_extractor,
                    device="cuda:0",
                )
                log.info("live ASR pipeline assembled in %.1fs (aoti=%s)",
                         time.perf_counter() - t0, self.aoti)
            except Exception:
                log.exception("live ASR assembly failed -- STT returns nothing")

    # -------------------------------------------------------------- config -- #
    def _gen_config(self):
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

        return OmniVoiceGenerationConfig(
            num_step=NUM_STEP,
            guidance_scale=GUIDANCE_SCALE,
            # We chunk at phrase level ourselves, so the internal splitter must
            # not fire: a threshold above any phrase we send keeps it off.
            audio_chunk_threshold=1e6,
            pad_duration=PAD_DURATION,
            fade_duration=FADE_DURATION,
        )

    # ---------------------------------------------------------------- voice -- #
    def set_voice(self, ref_bytes: bytes | None, ref_text: str | None,
                  key: str = "") -> dict:
        """Build (and cache) the voice-clone prompt from an uploaded clip.

        Costs ~750 ms, so it is done once per session rather than per phrase.
        ``ref_text`` may be None: OmniVoice transcribes the clip with its own
        Whisper pipe, which is already loaded here for live STT.
        """
        if ref_bytes is None:
            self._ensure_default_voice()
            return {"ok": True, "cloned": False}
        if key and key == self._voice_key and self._prompt is not None:
            return {"ok": True, "cloned": True, "cached": True}

        import soundfile as sf
        import torch

        t0 = time.perf_counter()
        wav, sr = sf.read(io.BytesIO(ref_bytes), dtype="float32", always_2d=False)
        wav = np.asarray(wav)
        if wav.ndim > 1:                     # stereo -> mono
            wav = wav.mean(axis=1)
        if wav.shape[0] > REF_MAX_SECS * sr:
            wav = wav[: int(REF_MAX_SECS * sr)]

        text = (ref_text or "").strip()
        if not text:
            text = self.transcribe_array(wav, sr)
            log.info("reference clip auto-transcribed: %d chars", len(text))

        # Record exactly what conditions the clone (presets and uploads reach
        # here by different routes).
        import hashlib
        log.info("clone inputs: %.1fs @%dHz sha=%s ref_text(%d)=%r",
                 wav.shape[0] / sr, int(sr),
                 hashlib.sha1(wav.tobytes()).hexdigest()[:10],
                 len(text), text[:120])
        self._prompt = self.model.create_voice_clone_prompt(
            (torch.from_numpy(wav)[None], int(sr)), ref_text=text or None)
        self._voice_key = key or None
        secs = time.perf_counter() - t0
        log.info("voice clone prompt built in %.0f ms (%.1fs of reference)",
                 secs * 1e3, wav.shape[0] / sr)
        return {"ok": True, "cloned": True, "ref_text": text,
                "seconds": round(secs, 3)}

    def _ensure_default_voice(self) -> None:
        """Pin ONE voice for the session when the user uploaded no clip.

        Without this the avatar changes voice mid-sentence. OmniVoice with no
        reference invents a speaker per generation, and we generate one phrase
        per call, so every phrase would be a different person. Upstream's own
        no-reference path has the same problem and solves it the same way:
        generate once, then clone from that audio for everything after.
        """
        if self._prompt is not None:
            return
        import torch

        seed = "Hey, good to see you. Let's talk."
        try:
            t0 = time.perf_counter()
            audio = self.model.generate(text=seed,
                                        generation_config=self._gen_config())
            a = np.asarray(audio[0], dtype=np.float32).reshape(-1)
            self._prompt = self.model.create_voice_clone_prompt(
                (torch.from_numpy(a)[None], TTS_SR), ref_text=seed)
            self._voice_key = "__default__"
            log.info("default voice pinned in %.0f ms",
                     (time.perf_counter() - t0) * 1e3)
        except Exception:
            log.exception("could not pin a default voice; each phrase may differ")

    @property
    def has_voice(self) -> bool:
        return self._prompt is not None

    # ------------------------------------------------------------------ tts -- #
    def split_phrases(self, text: str) -> list[str]:
        """Punctuation split sized for PHRASE_TARGET_SECS.

        Exposed so the parent can split without a round trip; the chunker is
        pure text handling and needs no GPU.
        """
        from omnivoice.utils.text import chunk_text_punctuation

        # ~15 chars per second of speech, the rate the library's own default
        # implies.
        chunk_len = max(24, int(PHRASE_TARGET_SECS * 15))
        try:
            return [c for c in chunk_text_punctuation(
                text, chunk_len=chunk_len, min_chunk_len=3) if c.strip()]
        except Exception:
            log.exception("phrase split failed; synthesising whole")
            return [text]

    def synth(self, text: str) -> dict:
        """Synthesise ONE phrase. Returns protocol-ready 16 kHz int16 PCM.

        Returning encoded PCM rather than float32 halves what crosses the fork
        and saves the parent a conversion on the hot path.
        """
        text = (text or "").strip()
        if not text:
            return {"pcm": b"", "ms": 0.0, "secs": 0.0}
        if self._prompt is None:      # nothing configured this session
            self._ensure_default_voice()
        t0 = time.perf_counter()
        audio = self.model.generate(
            text=text,
            voice_clone_prompt=self._prompt,
            generation_config=self._gen_config(),
        )
        a = np.asarray(audio[0], dtype=np.float32).reshape(-1)
        pcm = _to_wire_pcm(a)
        return {"pcm": pcm,
                "ms": (time.perf_counter() - t0) * 1e3,
                "secs": len(a) / TTS_SR}

    # ------------------------------------------------------------------ stt -- #
    def transcribe_array(self, wav_f32: np.ndarray, sr: int) -> str:
        try:
            return self.model.transcribe((np.asarray(wav_f32, dtype=np.float32), int(sr)))
        except Exception:
            log.exception("transcribe failed")
            return ""

    def transcribe_bytes(self, data: bytes) -> str:
        """Transcribe an uploaded clip (any container ffmpeg reads).

        Only the first ``REF_MAX_SECS`` are transcribed — the same span the
        voice clone uses, so the transcript describes what the clone hears.
        """
        pipe = getattr(self.model, "_asr_pipe", None)
        if pipe is None:
            return ""
        try:
            from transformers.pipelines.audio_utils import ffmpeg_read

            wav = ffmpeg_read(data, WIRE_SR)[: int(REF_MAX_SECS * WIRE_SR)]
            out = pipe(wav)
            return ((out or {}).get("text") or "").strip()
        except Exception:
            log.exception("upload transcription failed")
            return ""

    def transcribe(self, pcm_i16: bytes, sr: int = WIRE_SR) -> dict:
        """Live STT over one VAD-delimited utterance of 16 kHz int16 PCM."""
        if not pcm_i16:
            return {"text": "", "ms": 0.0}
        t0 = time.perf_counter()
        wav = np.frombuffer(pcm_i16, dtype="<i2").astype(np.float32) / 32768.0
        text = self.transcribe_array(wav, sr)
        return {"text": text, "ms": (time.perf_counter() - t0) * 1e3}

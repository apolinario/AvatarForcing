#!/usr/bin/env python
"""Real-time conversational avatar — dedicated-GPU entry point (port 7860).

Runs anywhere with a CUDA GPU: a local box, a cloud instance, or a paid HF
Space. Everything loads at boot; a session starts instantly and runs until the
visitor leaves (or ``AVATAR_SESSION_SECONDS``, if set).
"""
import logging, os, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

PORT   = int(os.environ.get("PORT", "7860"))
HOST   = os.environ.get("HOST", "0.0.0.0")
# The default avatar, used when a client sends no reference of its own. The UI
# preselects the same portrait, so this only matters for API-style callers.
REF    = os.environ.get("AVATAR_REF_IMAGE",
                        os.path.join(HERE, "static", "presets", "pearl.jpg"))
# The streaming server lives inside the model repo, so the repo root is HERE.
REPO   = os.environ.get("AVATAR_REPO_DIR", HERE)
DEVICE = os.environ.get("AVATAR_DEVICE", "cuda")
VISION = os.environ.get("AVATAR_VISION", "1").lower() not in ("0", "false", "no", "")
_ns    = os.environ.get("AVATAR_NORM_STD", "0.11").lower()
NORM   = None if _ns in ("", "auto", "none") else float(_ns)
FETCH  = os.environ.get("AVATAR_FETCH_WEIGHTS", "1").lower() not in ("0", "false", "no", "")

# --------------------------------------------------------------------------- #
# cold-start weights
#
# The Space's disk is wiped when it sleeps, so the checkpoints have to be back
# on disk before AvatarEngine.load() reads them (build_app / preload_blocking).
# The authors only distribute the two .pth files via Google Drive
# (AvatarForcing/download_weights.sh), which rate-limits large files; they are
# mirrored on the Hub instead. Do not confuse with lycui/AvatarForcing — that is
# a different model (arXiv 2603.14331) with different checkpoints.
# --------------------------------------------------------------------------- #
PRETRAINED = os.path.join(REPO, "pretrained_dir")
WAV2VEC    = os.path.join(PRETRAINED, "wav2vec2-base-960h")   # configs/inference.yaml: wav2vec_model_path
MIRROR     = os.environ.get("AVATAR_WEIGHTS_REPO", "multimodalart/AvatarForcingHelpers")
CKPTS      = ("motion_autoencoder.pth", "flow_transformer.pth")   # 181 MB + 614 MB


def _present(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def ensure_weights() -> None:
    """Fetch missing checkpoints into ``pretrained_dir`` (idempotent, ~1.1 GB cold).

    pretrained_dir ships only the empty ``checkpoint_here`` placeholder, which is
    why the presence test is per-file and size-based.
    """
    os.makedirs(PRETRAINED, exist_ok=True)

    for name in CKPTS:
        dst = os.path.join(PRETRAINED, name)
        if _present(dst):
            continue
        from huggingface_hub import hf_hub_download

        log.info("fetching %s from %s", name, MIRROR)
        t0 = time.perf_counter()
        try:
            hf_hub_download(MIRROR, name, local_dir=PRETRAINED)
        except Exception as exc:
            raise RuntimeError(
                f"could not download {name} from {MIRROR} to {dst}. Retry, or fetch "
                f"it manually with `hf download {MIRROR} {name} --local-dir "
                f"{PRETRAINED}` and restart."
            ) from exc
        log.info("fetched %s in %.1f s", name, time.perf_counter() - t0)

    # from_pretrained(local_files_only=True) needs config.json + the torch weights
    if not (_present(os.path.join(WAV2VEC, "config.json"))
            and any(_present(os.path.join(WAV2VEC, w))
                    for w in ("model.safetensors", "pytorch_model.bin"))):
        from huggingface_hub import snapshot_download

        log.info("fetching facebook/wav2vec2-base-960h -> %s", WAV2VEC)
        t0 = time.perf_counter()
        try:
            snapshot_download(
                "facebook/wav2vec2-base-960h",
                local_dir=WAV2VEC,
                ignore_patterns=["*.h5", "*.msgpack", "*.ot"],   # TF/Flax copies, ~1.5 GB unused
            )
        except Exception as exc:
            raise RuntimeError(
                f"could not download facebook/wav2vec2-base-960h to {WAV2VEC}: {exc}"
            ) from exc
        log.info("fetched wav2vec2-base-960h in %.1f s", time.perf_counter() - t0)


if FETCH:                                   # AVATAR_FETCH_WEIGHTS=0 for offline runs
    ensure_weights()

from server.conversation import DEFAULT_SYSTEM_PROMPT, ConversationBrain
from server.engine import AvatarEngine, FaceCropper
from server.gpu_session import _pin_cudnn_benchmark_off
from server.turn_detect import DETECTOR as TURN_DETECTOR
from server.web import build_app


# --------------------------------------------------------------------------- #
# semantic end-of-turn (server/turn_detect.py)
#
# Loaded here so the first person to pause does not pay for the download and the
# session build. Safe in this process precisely because it is onnxruntime and
# not torch: it never touches the CUDA that the comment above warns about.
# --------------------------------------------------------------------------- #
if FETCH:
    t0 = time.perf_counter()
    if TURN_DETECTOR.load():
        log.info("smart-turn ready in %.1f s", time.perf_counter() - t0)

# --------------------------------------------------------------------------- #
# models, loaded once at boot
# --------------------------------------------------------------------------- #
# face_alignment re-enables cudnn.benchmark on every detection, which triggers
# multi-second autotunes on new input shapes; pin it off for the process.
_pin_cudnn_benchmark_off()

_ENGINE = AvatarEngine(REF, repo_dir=REPO, device=DEVICE, seed=25,
                       avatar_norm_std=NORM, user_norm_std=None)
_CROPPER = None
if FETCH:
    t0 = time.perf_counter()
    _ENGINE.load_weights()
    _ENGINE.warm()
    log.info("avatar engine ready in %.1f s", time.perf_counter() - t0)
    _CROPPER = FaceCropper(device=DEVICE)


# --------------------------------------------------------------------------- #
# OmniVoice (TTS) + Whisper large-v3-turbo (STT), on the same GPU.
# --------------------------------------------------------------------------- #
TTS_REPO = os.environ.get("OMNIVOICE_REPO", "k2-fsa/OmniVoice")
ASR_REPO = os.environ.get("WHISPER_REPO", "openai/whisper-large-v3-turbo")
_ASR_MODEL = None
_ASR_PROC = None
_TTS = None

_SPEECH = None
if FETCH:
    import torch as _torch
    from omnivoice import OmniVoice
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    from server.speech import Speech

    t0 = time.perf_counter()
    _TTS = OmniVoice.from_pretrained(TTS_REPO, device_map="cuda:0",
                                     dtype=_torch.float16)
    _ASR_PROC = AutoProcessor.from_pretrained(ASR_REPO)
    _ASR_MODEL = WhisperForConditionalGeneration.from_pretrained(
        ASR_REPO, dtype=_torch.float16).to("cuda:0")
    _SPEECH = Speech(_TTS, asr_model=_ASR_MODEL, asr_processor=_ASR_PROC)
    log.info("speech (OmniVoice + Whisper) ready in %.1f s",
             time.perf_counter() - t0)




def speech_factory():
    return _SPEECH


def brain_factory(on_event, cfg, proxy=None):
    """One brain per WS session, carrying that session's UI overrides.

    The voice clip is cloned into the GPU worker by the web layer before
    priming; the brain just asks it to speak. ``cfg["system_prompt"]`` falsy ->
    the built-in prompt.
    """
    return ConversationBrain(
        on_event,
        proxy=proxy,
        use_vision=VISION,
        system_prompt=cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
    )


def voice_transcriber(audio_bytes: bytes) -> str:
    """Transcribe an uploaded voice clip on the resident GPU Whisper."""
    if _SPEECH is None:
        return ""
    return _SPEECH.transcribe_bytes(audio_bytes)


app = build_app(
    engine_factory=lambda: _ENGINE,
    brain_factory=brain_factory,
    face_cropper_factory=lambda: _CROPPER,
    speech_factory=speech_factory,
    voice_transcriber=voice_transcriber,
    default_system_prompt=DEFAULT_SYSTEM_PROMPT,
)

if __name__ == "__main__":
    # SO_REUSEADDR: uvicorn sets it, so only a live listener really blocks us
    # (without it, TIME_WAIT sockets from the previous run cause false failures).
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((HOST, PORT))
    except OSError as exc:
        raise SystemExit(f"[app] port {PORT} is not free: {exc}")
    finally:
        s.close()

    app.launch(
        server_name=HOST,        # "0.0.0.0" — required inside the Space
        server_port=PORT,        # 7860
        ssr_mode=False,
        show_error=True,
        quiet=False,
        share=False,
    )

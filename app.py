#!/usr/bin/env python
"""Real-time conversational avatar — HF Space main app (port 7860), ZeroGPU.

``import spaces`` is the FIRST import on purpose: it monkey-patches
``torch.cuda.*`` and installs the function mode that intercepts ``.to("cuda")``,
and it can only do that before torch initialises CUDA. Everything downstream
(``server.engine`` -> ``models.avatarforcing`` -> torch) depends on it having
run first.
"""
import spaces  # noqa: F401  # MUST precede torch / any CUDA-touching import

import logging, os, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

PORT   = int(os.environ.get("PORT", "7860"))
HOST   = os.environ.get("HOST", "0.0.0.0")
# Defaults to the reference portrait that ships with this repo. Point
# AVATAR_REF_IMAGE at your own photo (or upload one in the UI) to change it.
REF    = os.environ.get("AVATAR_REF_IMAGE", os.path.join(HERE, "data", "rumi.jpg"))
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


if FETCH:                                   # AVATAR_FETCH_WEIGHTS=0 for mock / offline runs
    ensure_weights()

from server.conversation import DEFAULT_SYSTEM_PROMPT, ConversationBrain
from server.cpu_asr import CPUTranscriber
from server.engine import AvatarEngine, FaceCropper
from server.web import build_app

# --------------------------------------------------------------------------- #
# CPU transcription child (POST /voice)
#
# Started HERE, before any model is built, for two reasons: the fork stays cheap,
# and — the important one — transcription must never run in this process. A
# transformers pipeline generating in the web process initialises CUDA (ZeroGPU
# reports cuda as available here), which poisons every later @spaces.GPU fork
# with "No CUDA GPUs are available". See server/cpu_asr.py.
# --------------------------------------------------------------------------- #
CPU_ASR_REPO = os.environ.get("WHISPER_CPU_REPO", "openai/whisper-small")
_CPU_ASR = CPUTranscriber(CPU_ASR_REPO) if FETCH else None

# --------------------------------------------------------------------------- #
# module-scope weight load (ZeroGPU packing)
#
# This builds the graph and moves every parameter to "cuda" in the MAIN process,
# where ZeroGPU intercepts the move: the real storage stays on CPU behind a fake
# CUDA alias and is packed so the first @spaces.GPU entry restores it straight
# into VRAM. Doing it here rather than in the fork is what keeps the ~37 s model
# load off the visitor's 90 s lease.
#
# load_weights() runs no forward pass -- that is the rule that makes it legal
# here. The reference latents and the cuDNN/cuBLAS warm-up need a real GPU and
# live in AvatarEngine.warm(), which server/gpu_session.py calls inside the
# lease. A forward pass at this point would silently execute on CPU.
# --------------------------------------------------------------------------- #
_ENGINE = AvatarEngine(REF, repo_dir=REPO, device=DEVICE, seed=25,
                       avatar_norm_std=NORM, user_norm_std=None)
if FETCH:
    t0 = time.perf_counter()
    log.info("building AvatarForcing + packing weights for ZeroGPU")
    _ENGINE.load_weights()
    log.info("weights ready in %.1f s", time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# OmniVoice (TTS) + its Whisper pipe (STT), replacing ElevenLabs
#
# Built at module scope for the same reason as the avatar engine: ZeroGPU
# intercepts the move to "cuda" here and packs the weights (~2.0 GB), so the
# first @spaces.GPU entry restores them into VRAM instead of paying a cold load
# out of the visitor's 90 s lease.
#
# There are deliberately TWO Whisper models, on different devices, because the
# two transcription jobs have opposite constraints:
#
#   * LIVE STT runs once per user utterance and the reply waits on it, so it has
#     to be fast -> large-v3-turbo on the GPU (~200-300 ms), packed with
#     everything else. On the Space's 2 vCPU it would take *seconds*, which is
#     not a conversation.
#   * The UPLOADED VOICE CLIP is transcribed once, before any lease exists, only
#     to condition the voice clone -> a small model on the CPU is plenty, and
#     running it here means the visitor spends no GPU quota to see the
#     transcript of their own clip.
# --------------------------------------------------------------------------- #
TTS_REPO = os.environ.get("OMNIVOICE_REPO", "k2-fsa/OmniVoice")
ASR_REPO = os.environ.get("WHISPER_REPO", "openai/whisper-large-v3-turbo")
_ASR_MODEL = None
_ASR_PROC = None
_TTS = None

if FETCH:
    import torch as _torch
    from omnivoice import OmniVoice

    t0 = time.perf_counter()
    log.info("building OmniVoice + packing weights for ZeroGPU")
    # load_asr is deliberately NOT set here. Building an HF pipeline with
    # device="cuda" in THIS process initialises a real CUDA context, which
    # poisons the ZeroGPU fork -- every @spaces.GPU call then dies in
    # worker_init with "No CUDA GPUs are available". The live ASR model is
    # loaded inside the lease instead (server/speech.py).
    _TTS = OmniVoice.from_pretrained(TTS_REPO, device_map="cuda:0",
                                     dtype=_torch.float16)
    log.info("OmniVoice ready in %.1f s", time.perf_counter() - t0)

    # Live-STT weights, loaded and packed here rather than read from disk inside
    # every lease. Only the weights: assembling the transformers *pipeline* is
    # what initialises CUDA in this process and poisons the fork, so that part
    # happens in the worker (server/speech.py).
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    t0 = time.perf_counter()
    _ASR_PROC = AutoProcessor.from_pretrained(ASR_REPO)
    _ASR_MODEL = WhisperForConditionalGeneration.from_pretrained(
        ASR_REPO, dtype=_torch.float16).to("cuda:0")
    log.info("live ASR weights ready in %.1f s", time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Pay AoTI's fixed start-up costs HERE, in the web process, so they do not come
# out of the visitor's 90 s session.
#
#  * valid_vec_isa_list() compiles a small CPU probe to detect the vector ISA.
#    spaces calls it from LazyAOTIModel.__init__, and it was the bulk of a
#    23.7 s in-lease "AoTI load". It is CPU-only, so it is safe here, and the
#    fork inherits the cached result.
#  * the package itself is a couple of MB; fetching it now means the lease only
#    maps it.
# --------------------------------------------------------------------------- #
AOTI_REPO = os.environ.get("OMNIVOICE_AOTI_REPO", "multimodalart/omnivoice-aoti")

if FETCH and AOTI_REPO and _TTS is not None:
    from server.speech import aoti_loader

    t0 = time.perf_counter()
    try:
        # All of this is CPU work: download the package, compile the inductor
        # vec-ISA probe, and bind the artifact to the module's parameter
        # tensors. The binding holds REFERENCES to those tensors, and ZeroGPU's
        # unpacking rebinds their .data to real VRAM in the worker -- so the
        # weights the artifact sees are the restored ones, with nothing
        # re-transferred. Doing it in the lease cost 23.7 s of the session.
        spaces.aoti_load(_TTS, repo_id=AOTI_REPO, aoti_loader=aoti_loader)
        log.info("AoTI package bound in %.1f s", time.perf_counter() - t0)
    except Exception:
        log.exception("AoTI bind failed -- synthesis will run eager")


def speech_factory():
    from server.speech import Speech

    return Speech(_TTS, asr_model=_ASR_MODEL, asr_processor=_ASR_PROC)


def brain_factory(on_event, cfg):
    """One brain per WS session, carrying that session's UI overrides.

    The voice itself is no longer the brain's business: the clip is cloned into
    the GPU worker by the web layer before priming, and the brain just asks it
    to speak. ``cfg["system_prompt"]`` falsy -> the built-in prompt.
    """
    return ConversationBrain(
        on_event,
        use_vision=VISION,
        system_prompt=cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
    )


def voice_transcriber(audio_bytes: bytes) -> str:
    """Whisper on the CPU for POST /voice — in a CHILD process, never here.

    Runs without a GPU lease, so the visitor uploads a clip, sees its transcript
    and only then starts a session, none of it billed to their quota. The child
    is not an optimisation: torch inference in this process initialises CUDA and
    breaks every later lease (server/cpu_asr.py).
    """
    if _CPU_ASR is None:
        return ""
    return _CPU_ASR.transcribe(audio_bytes)


app = build_app(
    # The already-packed engine, not a fresh one: the fork inherits this object
    # with its weights, and only warm() still has to run inside the lease.
    engine_factory=lambda: _ENGINE,
    brain_factory=brain_factory,
    face_cropper_factory=lambda: FaceCropper(device=DEVICE),
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

    # No preload step: the weights were built and packed at import above, and
    # the warm-up needs a GPU, which only exists inside a /run_session lease.
    app.launch(
        server_name=HOST,        # "0.0.0.0" — required inside the Space
        server_port=PORT,        # 7860
        ssr_mode=False,          # MUST override the Space's GRADIO_SSR_MODE=True
        show_error=True,
        quiet=False,
        share=False,
    )

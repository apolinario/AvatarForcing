#!/usr/bin/env python
"""Dev entry point for the real-time conversational avatar.

    MOCK=1 python dev_app.py          # default: mock engine + mock brain, no GPU
    MOCK=0 python dev_app.py          # real server.engine / server.conversation
    PORT=7871 python dev_app.py       # override the port

The final ``app.py`` (written by the orchestrator at integration time) uses the
exact same shape::

    from server.web import build_app
    app = build_app(engine_factory, brain_factory, cropper_factory)
    app.preload_blocking()                       # ~36 s model load, MOCK=0 only
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)

Environment knobs (all optional; defaults are the production ones)
-----------------------------------------------------------------
  MOCK                    1     use the mocks (no GPU / no API keys)
  PORT                    7861  http port
  HOST                    0.0.0.0
  AVATAR_REF_IMAGE  <app>/photo_2.jpeg   the avatar's reference photo
                          (legacy alias: REF_IMAGE)
  AVATAR_REPO_DIR   <app>/AvatarForcing  (legacy alias: REPO_DIR)
  AVATAR_DEVICE           cuda
  AVATAR_VISION           1     -> ConversationBrain(use_vision=...) : attach the
                                latest webcam snapshot to every user turn
  AVATAR_NORM_STD         0.11  fixed avatar-audio normalisation std. Measured
                                level of real ElevenLabs `eleven_flash_v2_5`
                                pcm_16000 output (see reports/integration.md);
                                "auto" -> engine's adaptive level tracker.
  AVATAR_USER_NORM_STD    auto  same for the mic stream (adaptive by default:
                                webcam mic gain is unknown and varies)
  AVATAR_SEED             25    per-session torch seed ("none" -> global RNG)
  AVATAR_PRELOAD          1     load the model before binding/serving
  AVATAR_JPEG_QUALITY     72    outbound frame JPEG quality      (server/web.py)
  AVATAR_JPEG_SUBSAMPLING 2     4:2:0                            (server/web.py)
  AVATAR_REPRIME_SECS     40    re-prime the engine this often (during a silent,
                                non-speaking block only) to reset generation
                                drift; without it, measured sharpness fell to
                                42 % by t=120 s. 0 disables.  (server/web.py)
  AVATAR_REPRIME_XFADE    10    frames of cross-dissolve hiding the re-prime pose
                                cut (0 = hard cut)            (server/web.py)
  AVATAR_CAM_WARM_SIZE    384   must match index.html's CAM_SIZE; warms the SFD
                                detector's cuDNN autotune (14 s!) at load time
  AVATAR_CONFIG_TIMEOUT   3.0   how long a session waits for the client's
                                {"type":"session_config"} (user voice id + the
                                optional uploaded reference photo) before falling
                                back to the server defaults      (server/web.py)
  AVATAR_MAX_REF_BYTES    8388608  cap on an uploaded reference (server/web.py)
  AVATAR_MAX_PROMPT_CHARS 8192  cap on a user-edited LLM system prompt, after
                                trimming                         (server/web.py)
  AVATAR_LEAD_BLOCKS      1.0   how far ahead of the wall clock to generate
  AVATAR_SNAPSHOT_PERIOD  1.0   seconds between brain.set_user_snapshot calls
  AVATAR_PACING_LOG_BLOCKS 25   pacing telemetry period (0 = off)

Process safety: this script only ever binds its own port and never touches other
processes. Stop it with ``.claude/hooks/safekill <pid>``.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _env_float_or_none(name: str, default: str) -> float | None:
    raw = os.environ.get(name, default).strip().lower()
    if raw in ("", "auto", "none", "adaptive"):
        return None
    return float(raw)


MOCK = _env_flag("MOCK", "1")
PORT = int(os.environ.get("PORT", "7861"))
HOST = os.environ.get("HOST", "0.0.0.0")
REF_IMAGE = os.environ.get("AVATAR_REF_IMAGE") or os.environ.get(
    "REF_IMAGE", os.path.join(HERE, "photo_2.jpeg"))
REPO_DIR = os.environ.get("AVATAR_REPO_DIR") or os.environ.get(
    "REPO_DIR", os.path.join(HERE, "AvatarForcing"))
DEVICE = os.environ.get("AVATAR_DEVICE", "cuda")
USE_VISION = _env_flag("AVATAR_VISION", "1")
AVATAR_NORM_STD = _env_float_or_none("AVATAR_NORM_STD", "0.11")
USER_NORM_STD = _env_float_or_none("AVATAR_USER_NORM_STD", "auto")
_seed_raw = os.environ.get("AVATAR_SEED", "25").strip().lower()
SEED = None if _seed_raw in ("", "none", "null") else int(_seed_raw)
PRELOAD = _env_flag("AVATAR_PRELOAD", "1")

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("dev_app")


def _assert_port_free(host: str, port: int) -> None:
    """DESIGN.md requires a bind check before assuming a port is free.

    SO_REUSEADDR is essential here: uvicorn sets it, so a *listener* is the only
    thing that can actually block the launch. Without it, the TIME_WAIT sockets
    left behind by the previous session's websocket make this check fail for ~60 s
    after every restart even though the port is perfectly bindable.
    """
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
    except OSError as exc:
        raise SystemExit(
            f"[dev_app] port {port} is NOT free ({exc}).\n"
            f"           Do NOT kill the owner blindly -- the 7860 demo's internal\n"
            f"           uvicorn also binds 7861. Re-run with e.g. PORT=7871.\n"
        ) from exc
    finally:
        s.close()


def _build_mock():
    from server.mock_conversation import MockConversationBrain
    from server.mock_engine import MockAvatarEngine, MockFaceCropper
    from server.web import build_app

    # The prompt editor must open on the *real* default even in MOCK mode, and
    # conversation.py's module scope is import-safe (numpy + optional dotenv, no
    # keys, no network). Falls back to "" if that ever stops being true.
    try:
        from server.conversation import DEFAULT_SYSTEM_PROMPT
    except Exception:
        DEFAULT_SYSTEM_PROMPT = ""

    log.info("MOCK mode: MockAvatarEngine + MockConversationBrain (ref=%s)", REF_IMAGE)
    return build_app(
        engine_factory=lambda: MockAvatarEngine(REF_IMAGE, repo_dir=REPO_DIR, device="cpu"),
        brain_factory=lambda on_event, cfg: MockConversationBrain(
            on_event, voice_id=cfg.get("voice_id"),
            system_prompt=cfg.get("system_prompt")),
        face_cropper_factory=lambda: MockFaceCropper(),
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


def _build_real():
    from server.conversation import DEFAULT_SYSTEM_PROMPT, ConversationBrain  # Agent D
    from server.engine import AvatarEngine, FaceCropper  # Agent C
    from server.web import build_app

    log.info(
        "REAL mode: AvatarEngine + ConversationBrain (ref=%s device=%s vision=%s "
        "avatar_norm_std=%s user_norm_std=%s seed=%s)",
        REF_IMAGE, DEVICE, USE_VISION, AVATAR_NORM_STD, USER_NORM_STD, SEED,
    )
    if not os.path.exists(REF_IMAGE):
        raise SystemExit(f"[dev_app] reference image not found: {REF_IMAGE}")

    def engine_factory():
        return AvatarEngine(
            REF_IMAGE,
            repo_dir=REPO_DIR,
            device=DEVICE,
            seed=SEED,
            avatar_norm_std=AVATAR_NORM_STD,
            user_norm_std=USER_NORM_STD,
        )

    def brain_factory(on_event, cfg):
        # One brain per WS session (conversation.md #5): one STT socket, one history.
        # cfg["voice_id"] falsy -> VOICE_ID env; cfg["system_prompt"] falsy ->
        # the built-in DEFAULT_SYSTEM_PROMPT.
        return ConversationBrain(
            on_event, use_vision=USE_VISION,
            voice_id=cfg.get("voice_id") or None,
            system_prompt=cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)

    def cropper_factory():
        # detector=None -> the module-cached shared SFD detector that
        # AvatarEngine.load() already built and warmed up.
        return FaceCropper(device=DEVICE)

    return build_app(
        engine_factory=engine_factory,
        brain_factory=brain_factory,
        face_cropper_factory=cropper_factory,
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


def main() -> None:
    _assert_port_free(HOST, PORT)
    app = _build_mock() if MOCK else _build_real()

    if PRELOAD:
        # engine.md caveat (a): load() blocks ~36 s -> pay it here, before the
        # first websocket is accepted, so /healthz is authoritative and the first
        # user only waits for start_session() (~240 ms).
        t0 = time.perf_counter()
        log.info("preloading engine (this takes ~36 s in REAL mode)...")
        app.preload_blocking()
        log.info("engine preloaded in %.1f s", time.perf_counter() - t0)

    log.info("launching on http://%s:%d  (MOCK=%s)", HOST, PORT, int(MOCK))
    # ssr_mode=False -> no node SSR sidecar; one process, one port, and our own
    # GET "/" stays in charge of the page.
    app.launch(
        server_name=HOST,
        server_port=PORT,
        ssr_mode=False,
        show_error=True,
        quiet=False,
        share=False,
    )


if __name__ == "__main__":
    main()

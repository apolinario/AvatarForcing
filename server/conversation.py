"""ConversationBrain — the voice conversation brain for the real-time avatar.

Pipeline
--------
    mic PCM (int16 @16 kHz)
        -> energy VAD (barge-in detection + fallback segmentation)
        -> ElevenLabs Scribe realtime STT websocket   (transport A, default)
           or ElevenLabs batch STT per VAD utterance  (transport B, fallback)
        -> final transcript
        -> LLM (HF router, OpenAI-compatible, streaming, vision: webcam snapshot)
        -> ElevenLabs TTS websocket "stream-input"    (transport A, default)
           or HTTP streaming TTS per sentence         (transport B, fallback)
        -> int16 16 kHz PCM appended to the avatar-audio buffer
        -> pull_avatar_audio() (consumed by the web layer / AvatarForcing engine)

Threading model
---------------
The brain owns a private asyncio event loop running on a daemon thread, so it is
completely decoupled from whatever loop (or plain sync code) creates it.
``feed_user_audio`` / ``set_user_snapshot`` / ``pull_avatar_audio`` are cheap,
non-blocking, lock-protected and safe to call from any thread.

See reports/conversation.md for measured latencies and configuration notes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import threading
import time
import typing as t
from dataclasses import dataclass, replace

import numpy as np

try:  # optional: only needed when the caller does not pre-load the environment
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

log = logging.getLogger("conversation")

SR = 16000
_BYTES_PER_SAMPLE = 2

DEFAULT_SYSTEM_PROMPT = (
    "You are the person in the photo, talking with someone over a live video call. "
    "You can see them through their webcam (an image of them is attached to their message) "
    "and you hear them speak. You are warm, curious, playful and completely natural.\n"
    "\n"
    "Hard rules for every reply:\n"
    "- Your words are spoken out loud by a text-to-speech voice. Plain conversational text ONLY: "
    "no markdown, no asterisks, no bullet points, no lists, no emoji, no stage directions, "
    "no parentheses describing actions.\n"
    "- Be CONCISE: one to three short sentences, the way a person actually talks. Never monologue.\n"
    "- Never mention that you are an AI, a model, or that you are looking at an image or a photo. "
    "You are simply seeing them on the call.\n"
    "- If their message is garbled or seems to be a transcription slip, just ask briefly what they meant.\n"
    "- Ask a short follow-up question when it keeps the conversation alive, but not every single turn."
)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class BrainConfig:
    """All tunable knobs. Override per-instance via ``ConversationBrain(..., **kwargs)``."""

    # --- credentials / endpoints (read from .env when None) ---
    eleven_api_key: t.Optional[str] = None      # ELEVEN_TOKEN
    hf_token: t.Optional[str] = None            # HF_TOKEN
    voice_id: t.Optional[str] = None            # VOICE_ID
    env_path: str = "/home/user/app/.env"

    # --- LLM ---
    llm_base_url: str = "https://router.huggingface.co/v1"
    llm_model: str = "google/gemma-4-31B-it:cerebras"
    llm_fallback_models: t.Tuple[str, ...] = (
        "google/gemma-4-31B-it",
        "google/gemma-3-27b-it",
    )
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 120
    temperature: float = 0.75
    history_max_messages: int = 16              # ~8 exchanges
    # Vision: attach the latest webcam snapshot to each user turn so the avatar can
    # "see" the user. When False, set_user_snapshot() is a no-op and no image is
    # ever sent (saves ~0.1 s of TTFT and the base64 payload).
    use_vision: bool = True
    # after this many consecutive image-related LLM failures, drop vision for the
    # rest of the session (0 = never give up)
    vision_disable_after_failures: int = 3

    # --- STT ---
    stt_transport: str = "realtime"             # "realtime" | "batch" | "off"
    stt_realtime_model: str = "scribe_v2_realtime"
    stt_batch_model: str = "scribe_v1"
    stt_language: t.Optional[str] = "en"
    stt_vad_silence_secs: float = 0.6           # server-side VAD commit threshold
    stt_mute_while_speaking: bool = False       # drop mic audio while avatar talks (anti-echo)
    stt_reconnect_delay: float = 1.0
    # wait this long after a final transcript for a follow-up commit and merge them
    # before calling the LLM (0 = start immediately; a split utterance then simply
    # cancels + restarts the turn, which costs one wasted LLM call)
    turn_debounce_ms: int = 0
    stt_max_utterance_secs: float = 25.0        # batch mode hard cap

    # --- TTS ---
    tts_transport: str = "websocket"            # "websocket" | "http"
    tts_model: str = "eleven_flash_v2_5"
    tts_stability: float = 0.45
    tts_similarity_boost: float = 0.8
    tts_speed: t.Optional[float] = None
    tts_auto_mode: bool = True
    tts_min_chunk_chars: int = 40               # phrase buffering before a ws send

    # --- local energy VAD ---
    vad_frame_ms: int = 20
    vad_start_factor: float = 3.0               # rms > noise_floor * factor -> speech
    vad_end_factor: float = 1.8
    vad_abs_floor: float = 220.0                # int16 rms; below this is never speech
    vad_min_speech_ms: int = 120
    vad_hangover_ms: int = 500

    # --- barge-in ---
    barge_in: bool = True
    barge_in_min_speech_ms: int = 280           # sustained speech needed to interrupt
    barge_in_grace_ms: int = 250                # ignore VAD right after we start speaking

    # --- buffers ---
    max_avatar_buffer_secs: float = 60.0
    max_input_backlog_secs: float = 30.0

    def resolved(self) -> "BrainConfig":
        """Fill credentials from the environment (never logged)."""
        if load_dotenv is not None and os.path.exists(self.env_path):
            load_dotenv(self.env_path, override=False)
        return replace(
            self,
            # Both spellings: the dedicated Space's secret is ELEVEN_TOKEN, the
            # ZeroGPU one's is ELEVENLABS_API_KEY. Accepting either keeps one
            # code path working against both Spaces' secret sets.
            eleven_api_key=(self.eleven_api_key
                            or os.environ.get("ELEVEN_TOKEN")
                            or os.environ.get("ELEVENLABS_API_KEY")
                            or ""),
            hf_token=self.hf_token or os.environ.get("HF_TOKEN") or "",
            voice_id=self.voice_id or os.environ.get("VOICE_ID") or "",
        )


# --------------------------------------------------------------------------- #
# energy VAD
# --------------------------------------------------------------------------- #
class EnergyVAD:
    """Frame-RMS voice activity detector with an adaptive noise floor + hangover.

    ``push()`` returns a list of ``("speech_start"|"speech_end", ms_of_speech)``
    transitions. The noise floor only adapts while *not* in speech so loud
    speech cannot drag the threshold up behind itself.
    """

    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg
        self.frame = max(1, int(SR * cfg.vad_frame_ms / 1000))
        self._tail = np.zeros(0, dtype=np.float32)
        self.noise_floor = cfg.vad_abs_floor / 2.0
        self.active = False
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self._cand_ms = 0.0
        self.last_rms = 0.0

    @property
    def thr_on(self) -> float:
        return max(self.cfg.vad_abs_floor, self.noise_floor * self.cfg.vad_start_factor)

    @property
    def thr_off(self) -> float:
        return max(self.cfg.vad_abs_floor * 0.6, self.noise_floor * self.cfg.vad_end_factor)

    def push(self, pcm: np.ndarray) -> t.List[t.Tuple[str, float]]:
        events: t.List[t.Tuple[str, float]] = []
        buf = np.concatenate([self._tail, pcm.astype(np.float32)]) if self._tail.size else pcm.astype(np.float32)
        n_frames = buf.size // self.frame
        self._tail = buf[n_frames * self.frame:].copy()
        if n_frames == 0:
            return events
        frames = buf[: n_frames * self.frame].reshape(n_frames, self.frame)
        rms = np.sqrt(np.maximum((frames * frames).mean(axis=1), 0.0))
        fm = self.cfg.vad_frame_ms

        for r in rms:
            self.last_rms = float(r)
            if not self.active:
                # adaptive floor: fast down, slow up
                alpha = 0.25 if r < self.noise_floor else 0.01
                self.noise_floor += alpha * (r - self.noise_floor)
                self.noise_floor = max(self.noise_floor, 1.0)
                if r > self.thr_on:
                    self._cand_ms += fm
                    if self._cand_ms >= self.cfg.vad_min_speech_ms:
                        self.active = True
                        self.speech_ms = self._cand_ms
                        self.silence_ms = 0.0
                        self._cand_ms = 0.0
                        events.append(("speech_start", self.speech_ms))
                else:
                    self._cand_ms = max(0.0, self._cand_ms - fm)
            else:
                if r > self.thr_off:
                    self.speech_ms += fm
                    self.silence_ms = 0.0
                else:
                    self.silence_ms += fm
                    if self.silence_ms >= self.cfg.vad_hangover_ms:
                        self.active = False
                        events.append(("speech_end", self.speech_ms))
                        self.speech_ms = 0.0
                        self._cand_ms = 0.0
        return events


# --------------------------------------------------------------------------- #
# text chunking for streaming TTS
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"[.!?;:]['\")\]]?\s")
_CLAUSE_END = re.compile(r"[,—-]\s")
_SPEAKABLE = re.compile(r"[A-Za-z0-9À-ɏ]")


def clean_for_speech(text: str) -> str:
    """Strip markdown/emoji-ish artifacts the LLM may still emit."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_`#>|]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_IMAGE_ERROR_HINTS = (
    "invalid_image", "image data", "'image'", '"image"', "image_url", "image too",
    "payload too large", "request entity too large", "413",
)


def _is_image_error(exc: BaseException) -> bool:
    """Does this API error look like the attached snapshot's fault?"""
    s = str(exc).lower()
    return any(h in s for h in _IMAGE_ERROR_HINTS)


def _has_image(messages: t.Sequence[dict]) -> bool:
    return any(
        isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
        for m in messages
    )


def _strip_images(messages: t.Sequence[dict]) -> t.List[dict]:
    """Same conversation, text only — used to retry a turn whose image was rejected."""
    out: t.List[dict] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            text = " ".join(p.get("text", "") for p in c if p.get("type") == "text").strip()
            out.append({"role": m["role"], "content": text})
        else:
            out.append(dict(m))
    return out


class PhraseChunker:
    """Accumulate LLM deltas, release speakable phrases at natural boundaries."""

    def __init__(self, min_chars: int = 40):
        self.min_chars = min_chars
        self.buf = ""

    def push(self, delta: str) -> t.List[str]:
        self.buf += delta
        out: t.List[str] = []
        while True:
            m = None
            for rx in (_SENTENCE_END, _CLAUSE_END):
                mm = rx.search(self.buf)
                if mm and (m is None or mm.end() < m.end()):
                    m = mm
            if m is None:
                break
            head, rest = self.buf[: m.end()], self.buf[m.end():]
            if len(head.strip()) < self.min_chars and rest.strip() == "":
                break  # wait for more text, might be a short fragment mid-stream
            out.append(head)
            self.buf = rest
        return [c for c in out if c.strip()]

    def flush(self) -> str:
        s, self.buf = self.buf, ""
        return s


# --------------------------------------------------------------------------- #
# ElevenLabs streaming TTS over websocket "stream-input"
# --------------------------------------------------------------------------- #
class _WebsocketTTS:
    """One websocket per avatar turn. Opened eagerly so connect latency
    overlaps the LLM's time-to-first-token."""

    def __init__(self, cfg: BrainConfig, on_pcm: t.Callable[[bytes], None]):
        self.cfg = cfg
        self.on_pcm = on_pcm
        self._ws = None
        self._reader: t.Optional[asyncio.Task] = None
        self.final = asyncio.Event()
        self.first_audio_at: t.Optional[float] = None
        self.opened_at: t.Optional[float] = None
        self.total_samples = 0
        self.error: t.Optional[str] = None

    def _url(self) -> str:
        params = [
            f"model_id={self.cfg.tts_model}",
            "output_format=pcm_16000",
            "inactivity_timeout=20",
        ]
        if self.cfg.tts_auto_mode:
            params.append("auto_mode=true")
        return (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self.cfg.voice_id}"
            f"/stream-input?" + "&".join(params)
        )

    async def open(self) -> None:
        from websockets.asyncio.client import connect as ws_connect

        t0 = time.monotonic()
        self._ws = await ws_connect(
            self._url(),
            additional_headers={"xi-api-key": self.cfg.eleven_api_key},
            max_queue=64,
            open_timeout=10,
        )
        self.opened_at = time.monotonic()
        log.debug("tts ws open in %.3fs", self.opened_at - t0)
        vs: t.Dict[str, t.Any] = {
            "stability": self.cfg.tts_stability,
            "similarity_boost": self.cfg.tts_similarity_boost,
        }
        if self.cfg.tts_speed is not None:
            vs["speed"] = self.cfg.tts_speed
        init: t.Dict[str, t.Any] = {"text": " ", "voice_settings": vs}
        if not self.cfg.tts_auto_mode:
            init["generation_config"] = {"chunk_length_schedule": [50, 120, 200, 300]}
        await self._ws.send(json.dumps(init))
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:  # type: ignore[union-attr]
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                audio = msg.get("audio")
                if audio:
                    pcm = base64.b64decode(audio)
                    if pcm:
                        if self.first_audio_at is None:
                            self.first_audio_at = time.monotonic()
                        self.total_samples += len(pcm) // _BYTES_PER_SAMPLE
                        self.on_pcm(pcm)
                if msg.get("error") or msg.get("message"):
                    self.error = str(msg.get("error") or msg.get("message"))
                    log.warning("tts ws message: %s", self.error)
                if msg.get("isFinal"):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # connection dropped
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("tts ws read error: %s", self.error)
        finally:
            self.final.set()

    async def send_text(self, text: str) -> None:
        if self._ws is None or not text:
            return
        await self._ws.send(json.dumps({"text": text, "try_trigger_generation": True}))

    async def finish(self, timeout: float = 30.0) -> None:
        """Signal end of input and wait for the last audio chunk."""
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"text": ""}))
            except Exception:
                pass
        try:
            await asyncio.wait_for(self.final.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("tts ws finish timed out")

    async def close(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


# --------------------------------------------------------------------------- #
# the brain
# --------------------------------------------------------------------------- #
class ConversationBrain:
    """Interface (see DESIGN.md):

        brain = ConversationBrain(on_event)                 # vision on (default)
        brain = ConversationBrain(on_event, use_vision=False)  # or any BrainConfig field
        await brain.start()
        brain.feed_user_audio(np.int16 pcm @16k mono)      # ~every 100 ms, any thread
        brain.set_user_snapshot(jpeg_bytes)                 # ~1/s, any thread
        pcm = brain.pull_avatar_audio(6400)                 # every 400 ms, any thread
        await brain.stop()
    """

    def __init__(self, on_event: t.Callable[[dict], None], config: t.Optional[BrainConfig] = None, **overrides: t.Any):
        cfg = config or BrainConfig()
        if overrides:
            cfg = replace(cfg, **overrides)
        self.cfg = cfg.resolved()
        self._on_event_cb = on_event

        # --- cross-thread state ---
        self._in_lock = threading.Lock()
        self._in_chunks: t.List[np.ndarray] = []
        self._in_samples = 0
        self._audio_lock = threading.Lock()
        self._avatar_pcm = bytearray()
        self._consumed_samples = 0
        self._tts_stream_done = True
        self._snap_lock = threading.Lock()
        self._snapshot: t.Optional[bytes] = None

        # --- loop-owned state ---
        self._loop: t.Optional[asyncio.AbstractEventLoop] = None
        self._thread: t.Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._loop_error: t.Optional[BaseException] = None
        self._running = False
        self._state = "listening"
        self._vad = EnergyVAD(self.cfg)
        self._history: t.List[dict] = []
        self._turn_task: t.Optional[asyncio.Task] = None
        self._turn_seq = 0
        self._turn_consumed_at_start = 0
        self._pending_user_text = ""
        self._speaking_started_at = 0.0
        self._turn_first_audio_at: t.Optional[float] = None
        self._turn_audio_samples = 0
        self._debounce_text = ""
        self._debounce_task: t.Optional[asyncio.Task] = None
        self._vision_disabled = False
        self._vision_fail_count = 0
        self._turn_vision_dropped = False
        self._audio_evt: t.Optional[asyncio.Event] = None
        self._wake: t.Optional[t.Callable[[], None]] = None

        # --- STT transport state ---
        self._stt_conn = None
        self._stt_task: t.Optional[asyncio.Task] = None
        self._stt_ready = False
        self._batch_buf = bytearray()
        self._batch_preroll = bytearray()
        self._batch_active = False
        self._eleven = None  # AsyncElevenLabs
        self._llm = None  # AsyncOpenAI
        self._llm_model = self.cfg.llm_model

        # --- instrumentation (used by the test harness / report) ---
        self.metrics: t.Dict[str, t.Any] = {"turns": []}
        self._last_speech_end_wall: t.Optional[float] = None

    # ------------------------------------------------------------------ #
    # public, thread-safe API
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main, name="conv-brain", daemon=True)
        self._thread.start()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._ready.wait(30)
        else:
            await loop.run_in_executor(None, self._ready.wait, 30)
        if self._loop_error is not None:
            raise self._loop_error

    async def stop(self) -> None:
        thread, loop = self._thread, self._loop
        self._thread = None
        if loop is not None and thread is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            except Exception:
                pass
            try:
                cur = asyncio.get_running_loop()
            except RuntimeError:
                thread.join(15)
            else:
                await cur.run_in_executor(None, thread.join, 15)
        self._loop = None

    def feed_user_audio(self, pcm: np.ndarray) -> None:
        """Non-blocking. ``pcm``: np.int16, 16 kHz, mono."""
        if pcm is None:
            return
        arr = np.asarray(pcm)
        if arr.dtype != np.int16:
            if arr.dtype in (np.float32, np.float64):
                arr = np.clip(arr, -1.0, 1.0) * 32767.0
            arr = arr.astype(np.int16)
        arr = arr.reshape(-1)
        if arr.size == 0:
            return
        cap = int(self.cfg.max_input_backlog_secs * SR)
        with self._in_lock:
            self._in_chunks.append(arr)
            self._in_samples += arr.size
            while self._in_samples > cap and len(self._in_chunks) > 1:
                self._in_samples -= self._in_chunks.pop(0).size
        wake = self._wake
        if wake is not None:
            wake()

    def set_user_snapshot(self, jpeg_bytes: bytes) -> None:
        """Store the latest webcam JPEG for the next user turn.
        Cheap no-op when ``use_vision`` is False (or vision got disabled)."""
        if not jpeg_bytes or not self.cfg.use_vision or self._vision_disabled:
            return
        with self._snap_lock:
            self._snapshot = bytes(jpeg_bytes)

    def pull_avatar_audio(self, n_samples: int) -> np.ndarray:
        """Return exactly ``n_samples`` int16 samples, silence-padded when idle.
        Consumption is monotonic — samples are removed from the buffer."""
        n = max(0, int(n_samples))
        out = np.zeros(n, dtype=np.int16)
        if n == 0:
            return out
        with self._audio_lock:
            avail = len(self._avatar_pcm) // _BYTES_PER_SAMPLE
            take = min(avail, n)
            if take:
                out[:take] = np.frombuffer(bytes(self._avatar_pcm[: take * _BYTES_PER_SAMPLE]), dtype=np.int16)
                del self._avatar_pcm[: take * _BYTES_PER_SAMPLE]
                self._consumed_samples += take
            drained = len(self._avatar_pcm) == 0
            done = self._tts_stream_done
        if drained and done and self._state == "speaking":
            self._call_soon(self._finish_speaking)
        return out

    # convenience for sync callers / tests
    def start_blocking(self, timeout: float = 30.0) -> None:
        if self._thread is not None:
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main, name="conv-brain", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        if self._loop_error is not None:
            raise self._loop_error

    def stop_blocking(self, timeout: float = 15.0) -> None:
        thread, loop = self._thread, self._loop
        self._thread = None
        if loop is not None and thread is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            except Exception:
                pass
            thread.join(timeout)
        self._loop = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def vision_active(self) -> bool:
        """True while snapshots are being attached to user turns."""
        return bool(self.cfg.use_vision) and not self._vision_disabled

    @property
    def active_transports(self) -> t.Dict[str, t.Any]:
        return {
            "stt": self.cfg.stt_transport,
            "tts": self.cfg.tts_transport,
            "llm_model": self._llm_model,
            "vision": self.vision_active,
        }

    # ------------------------------------------------------------------ #
    # loop plumbing
    # ------------------------------------------------------------------ #
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except BaseException as exc:  # pragma: no cover
            self._loop_error = exc
            log.exception("brain loop crashed")
        finally:
            self._ready.set()
            try:
                pending = [tk for tk in asyncio.all_tasks(loop) if not tk.done()]
                for tk in pending:
                    tk.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    def _call_soon(self, fn: t.Callable[[], t.Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(fn)
        except RuntimeError:
            pass

    def _emit(self, event: dict) -> None:
        try:
            self._on_event_cb(event)
        except Exception:
            log.exception("on_event callback raised")

    def _set_state(self, value: str) -> None:
        if value == self._state:
            return
        self._state = value
        if value == "speaking":
            self._speaking_started_at = time.monotonic()
        self._emit({"type": "state", "value": value})

    def _error(self, text: str) -> None:
        log.warning("brain error: %s", text)
        self._emit({"type": "error", "text": text})

    async def _main(self) -> None:
        self._running = True
        self._audio_evt = asyncio.Event()
        evt, loop = self._audio_evt, asyncio.get_running_loop()

        def wake() -> None:
            try:
                loop.call_soon_threadsafe(evt.set)
            except RuntimeError:
                pass

        self._wake = wake
        try:
            self._init_clients()
        except Exception as exc:
            self._error(f"client init failed: {type(exc).__name__}: {exc}")
        if self.cfg.stt_transport == "realtime":
            self._stt_task = asyncio.create_task(self._stt_supervisor())
        self._state = "listening"
        self._emit({"type": "state", "value": "listening"})  # always announce the initial state
        self._ready.set()
        try:
            await self._pump()
        finally:
            self._running = False

    def _init_clients(self) -> None:
        from elevenlabs.client import AsyncElevenLabs
        from openai import AsyncOpenAI

        if not self.cfg.eleven_api_key:
            raise RuntimeError("ELEVEN_TOKEN missing")
        if not self.cfg.hf_token:
            raise RuntimeError("HF_TOKEN missing")
        if not self.cfg.voice_id:
            raise RuntimeError("VOICE_ID missing")
        self._eleven = AsyncElevenLabs(api_key=self.cfg.eleven_api_key)
        self._llm = AsyncOpenAI(base_url=self.cfg.llm_base_url, api_key=self.cfg.hf_token, timeout=60.0)

    async def _shutdown(self) -> None:
        self._running = False
        await self._cancel_turn(flush_audio=False)
        if self._stt_task is not None:
            self._stt_task.cancel()
            try:
                await self._stt_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stt_task = None
        await self._close_stt()
        if self._audio_evt is not None:
            self._audio_evt.set()

    # ------------------------------------------------------------------ #
    # audio pump: VAD + STT feeding + barge-in
    # ------------------------------------------------------------------ #
    async def _pump(self) -> None:
        assert self._audio_evt is not None
        while self._running:
            try:
                await asyncio.wait_for(self._audio_evt.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass
            self._audio_evt.clear()
            if not self._running:
                break
            with self._in_lock:
                chunks, self._in_chunks, self._in_samples = self._in_chunks, [], 0
            if not chunks:
                continue
            pcm = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            try:
                await self._process_audio(pcm)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._error(f"audio pipeline: {type(exc).__name__}: {exc}")

    async def _process_audio(self, pcm: np.ndarray) -> None:
        events = self._vad.push(pcm)
        for kind, ms in events:
            if kind == "speech_start":
                log.debug("VAD speech_start (nf=%.0f thr=%.0f)", self._vad.noise_floor, self._vad.thr_on)
            else:
                self._last_speech_end_wall = time.monotonic()
                log.debug("VAD speech_end after %.0fms", ms)

        # barge-in: sustained local speech while the avatar is thinking/speaking
        if self.cfg.barge_in and self._vad.active and self._state in ("thinking", "speaking"):
            grace_ok = (
                self._state != "speaking"
                or (time.monotonic() - self._speaking_started_at) * 1000.0 >= self.cfg.barge_in_grace_ms
            )
            if grace_ok and self._vad.speech_ms >= self.cfg.barge_in_min_speech_ms:
                log.info("barge-in (%.0fms speech during %s)", self._vad.speech_ms, self._state)
                await self._cancel_turn(flush_audio=True)
                self._set_state("listening")

        muted = self.cfg.stt_mute_while_speaking and self._state == "speaking"
        if self.cfg.stt_transport == "realtime":
            if not muted:
                await self._stt_feed_realtime(pcm)
        elif self.cfg.stt_transport == "batch":
            if not muted:
                self._stt_feed_batch(pcm, events)

    # ------------------------------------------------------------------ #
    # STT transport A: Scribe realtime websocket
    # ------------------------------------------------------------------ #
    async def _stt_supervisor(self) -> None:
        """Keep the realtime STT websocket alive, reconnecting with backoff."""
        delay = self.cfg.stt_reconnect_delay
        while self._running:
            try:
                await self._open_stt()
                delay = self.cfg.stt_reconnect_delay
                while self._running and self._stt_ready:
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._error(f"STT connect failed: {type(exc).__name__}: {exc}")
            self._stt_ready = False
            await self._close_stt()
            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)

    async def _open_stt(self) -> None:
        from elevenlabs.realtime import AudioFormat, CommitStrategy, RealtimeEvents

        opts: t.Dict[str, t.Any] = {
            "model_id": self.cfg.stt_realtime_model,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": SR,
            "commit_strategy": CommitStrategy.VAD,
            "vad_silence_threshold_secs": self.cfg.stt_vad_silence_secs,
        }
        if self.cfg.stt_language:
            opts["language_code"] = self.cfg.stt_language
        conn = await self._eleven.speech_to_text.realtime.connect(opts)  # type: ignore[union-attr]
        self._stt_conn = conn
        self._stt_ready = True

        conn.on(RealtimeEvents.PARTIAL_TRANSCRIPT.value, self._on_stt_partial)
        conn.on(RealtimeEvents.COMMITTED_TRANSCRIPT.value, self._on_stt_committed)
        conn.on(RealtimeEvents.CLOSE.value, self._on_stt_close)
        for ev in (
            RealtimeEvents.AUTH_ERROR,
            RealtimeEvents.QUOTA_EXCEEDED,
            RealtimeEvents.TRANSCRIBER_ERROR,
            RealtimeEvents.RATE_LIMITED,
            RealtimeEvents.INPUT_ERROR,
            RealtimeEvents.RESOURCE_EXHAUSTED,
            RealtimeEvents.SESSION_TIME_LIMIT_EXCEEDED,
            RealtimeEvents.UNACCEPTED_TERMS_ERROR,
        ):
            conn.on(ev.value, self._on_stt_error)
        log.info("STT realtime websocket open (%s)", self.cfg.stt_realtime_model)

    async def _close_stt(self) -> None:
        conn, self._stt_conn = self._stt_conn, None
        self._stt_ready = False
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass

    async def _stt_feed_realtime(self, pcm: np.ndarray) -> None:
        conn = self._stt_conn
        if conn is None or not self._stt_ready:
            return
        try:
            await conn.send({"audio_base_64": base64.b64encode(pcm.tobytes()).decode("ascii")})
        except Exception as exc:
            log.warning("STT send failed (%s), will reconnect", exc)
            self._stt_ready = False

    def _on_stt_partial(self, data: dict | None = None) -> None:
        text = (data or {}).get("text") or ""
        if text.strip():
            self._emit({"type": "user_transcript", "text": text.strip(), "final": False})

    def _on_stt_committed(self, data: dict | None = None) -> None:
        text = (data or {}).get("text") or ""
        self._on_final_transcript(text)

    def _on_stt_close(self, *_: t.Any) -> None:
        if self._running:
            log.info("STT websocket closed")
        self._stt_ready = False

    def _on_stt_error(self, data: dict | None = None) -> None:
        d = data or {}
        msg = d.get("message") or d.get("error") or d.get("message_type") or "unknown STT error"
        self._error(f"STT: {msg}")
        if d.get("message_type") in ("auth_error", "session_time_limit_exceeded", "resource_exhausted"):
            self._stt_ready = False

    # ------------------------------------------------------------------ #
    # STT transport B: batch endpoint per VAD-segmented utterance
    # ------------------------------------------------------------------ #
    def _stt_feed_batch(self, pcm: np.ndarray, events: t.List[t.Tuple[str, float]]) -> None:
        raw = pcm.tobytes()
        started = any(k == "speech_start" for k, _ in events)
        ended = any(k == "speech_end" for k, _ in events)
        if started and not self._batch_active:
            self._batch_active = True
            self._batch_buf = bytearray(self._batch_preroll)  # keep 300 ms pre-roll
        if self._batch_active:
            self._batch_buf.extend(raw)
            cap = int(self.cfg.stt_max_utterance_secs * SR) * _BYTES_PER_SAMPLE
            if ended or len(self._batch_buf) >= cap:
                utt = bytes(self._batch_buf)
                self._batch_buf = bytearray()
                self._batch_active = False
                asyncio.create_task(self._batch_transcribe(utt))
        preroll_cap = int(0.3 * SR) * _BYTES_PER_SAMPLE
        self._batch_preroll.extend(raw)
        if len(self._batch_preroll) > preroll_cap:
            del self._batch_preroll[: len(self._batch_preroll) - preroll_cap]

    async def _batch_transcribe(self, pcm_bytes: bytes) -> None:
        if len(pcm_bytes) < int(0.25 * SR) * _BYTES_PER_SAMPLE:
            return
        try:
            res = await self._eleven.speech_to_text.convert(  # type: ignore[union-attr]
                model_id=self.cfg.stt_batch_model,
                file=("utterance.pcm", pcm_bytes, "application/octet-stream"),
                file_format="pcm_s16le_16",
                language_code=self.cfg.stt_language,
                tag_audio_events=False,
            )
            text = getattr(res, "text", None) or ""
        except Exception as exc:
            self._error(f"batch STT failed: {type(exc).__name__}: {exc}")
            return
        self._on_final_transcript(text)

    # ------------------------------------------------------------------ #
    # turn management
    # ------------------------------------------------------------------ #
    def _on_final_transcript(self, text: str) -> None:
        text = (text or "").strip()
        if not text or not _SPEAKABLE.search(text):
            return
        self._emit({"type": "user_transcript", "text": text, "final": True})
        loop = self._loop
        if loop is None:
            return
        if self.cfg.turn_debounce_ms > 0:
            coro = self._debounced_begin(text)
        else:
            coro = self._begin_turn(text)
        # STT callbacks already fire on the brain loop; batch STT may not
        try:
            asyncio.get_running_loop()
            asyncio.create_task(coro)
        except RuntimeError:
            asyncio.run_coroutine_threadsafe(coro, loop)

    async def _debounced_begin(self, text: str) -> None:
        self._debounce_text = (self._debounce_text + " " + text).strip()
        task, self._debounce_task = self._debounce_task, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
        self._debounce_task = asyncio.current_task()
        await asyncio.sleep(self.cfg.turn_debounce_ms / 1000.0)
        merged, self._debounce_text = self._debounce_text, ""
        self._debounce_task = None
        if merged:
            await self._begin_turn(merged)

    async def _begin_turn(self, text: str) -> None:
        merged = text
        if self._turn_task is not None and not self._turn_task.done():
            if self._state == "thinking":
                # user kept talking before we said anything -> merge, restart
                merged = (self._pending_user_text + " " + text).strip()
            await self._cancel_turn(flush_audio=True)
        self._pending_user_text = merged
        self._turn_seq += 1
        self._turn_task = asyncio.create_task(self._run_turn(merged, self._turn_seq))

    async def _cancel_turn(self, flush_audio: bool) -> None:
        task, self._turn_task = self._turn_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if flush_audio:
            self.flush_avatar_audio()

    def flush_avatar_audio(self) -> int:
        """Drop all un-consumed avatar audio (barge-in). Returns samples dropped."""
        with self._audio_lock:
            dropped = len(self._avatar_pcm) // _BYTES_PER_SAMPLE
            self._avatar_pcm.clear()
            self._tts_stream_done = True
        if dropped:
            log.info("flushed %d unconsumed avatar samples (%.2fs)", dropped, dropped / SR)
        return dropped

    def _append_avatar_pcm(self, pcm: bytes) -> None:
        cap = int(self.cfg.max_avatar_buffer_secs * SR) * _BYTES_PER_SAMPLE
        with self._audio_lock:
            self._avatar_pcm.extend(pcm)
            if len(self._avatar_pcm) > cap:
                del self._avatar_pcm[: len(self._avatar_pcm) - cap]
        if self._turn_first_audio_at is None:
            self._turn_first_audio_at = time.monotonic()
        self._turn_audio_samples += len(pcm) // _BYTES_PER_SAMPLE
        if self._state != "speaking":
            self._set_state("speaking")

    def _finish_speaking(self) -> None:
        if self._state == "speaking":
            self._set_state("listening")

    # ------------------------------------------------------------------ #
    def _build_messages(self, user_text: str) -> t.List[dict]:
        msgs: t.List[dict] = [{"role": "system", "content": self.cfg.system_prompt}]
        hist = self._history[-self.cfg.history_max_messages:]
        # merge consecutive same-role messages (can happen after a barge-in)
        for m in hist:
            if msgs and msgs[-1]["role"] == m["role"] and m["role"] != "system":
                msgs[-1] = {"role": m["role"], "content": f"{msgs[-1]['content']} {m['content']}".strip()}
            else:
                msgs.append(dict(m))
        snap = None
        if self.vision_active:
            with self._snap_lock:
                snap = self._snapshot
        if snap:
            url = "data:image/jpeg;base64," + base64.b64encode(snap).decode("ascii")
            content: t.Any = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        else:
            content = user_text
        if msgs and msgs[-1]["role"] == "user":
            # previous user turn never got a reply: fold it into the text part
            prev = msgs.pop()
            prev_text = prev["content"] if isinstance(prev["content"], str) else ""
            if prev_text:
                if isinstance(content, list):
                    content[0]["text"] = f"{prev_text} {user_text}".strip()
                else:
                    content = f"{prev_text} {user_text}".strip()
        msgs.append({"role": "user", "content": content})
        return msgs

    async def _open_llm_stream(self, messages: t.List[dict], allow_vision_retry: bool = True) -> t.Any:
        """Open a streaming chat completion.

        Falls back to the next configured model id if the preferred one is
        unavailable, and — if the request carried an image and every model failed —
        retries once with the image stripped, so a bad/oversized snapshot can never
        cost the user their turn.
        """
        candidates = [self._llm_model] + [m for m in self.cfg.llm_fallback_models if m != self._llm_model]
        last: t.Optional[BaseException] = None
        for cand in candidates:
            try:
                stream = await self._llm.chat.completions.create(  # type: ignore[union-attr]
                    model=cand,
                    messages=messages,
                    stream=True,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = exc
                log.warning("LLM model %s failed: %s", cand, exc)
                if allow_vision_retry and _has_image(messages) and _is_image_error(exc):
                    break  # the snapshot is the problem, not the model — retry without it
                continue
            if cand != self._llm_model:
                self._error(f"LLM model {self._llm_model} unavailable, switched to {cand}")
                self._llm_model = cand
            if _has_image(messages):
                self._vision_fail_count = 0
            return stream

        if allow_vision_retry and _has_image(messages):
            self._vision_fail_count += 1
            self._turn_vision_dropped = True
            self._error(
                f"LLM call with webcam snapshot failed ({type(last).__name__}: "
                f"{str(last)[:160]}); retrying this turn without vision"
            )
            if 0 < self.cfg.vision_disable_after_failures <= self._vision_fail_count:
                self._vision_disabled = True
                with self._snap_lock:
                    self._snapshot = None
                self._error(
                    f"vision disabled for this session after {self._vision_fail_count} "
                    "consecutive image failures"
                )
            return await self._open_llm_stream(_strip_images(messages), allow_vision_retry=False)

        assert last is not None
        raise last

    async def _run_turn(self, user_text: str, seq: int) -> None:
        t_turn = time.monotonic()
        speech_end = self._last_speech_end_wall
        m: t.Dict[str, t.Any] = {
            "seq": seq,
            "user_text": user_text,
            "t_turn_start_monotonic": t_turn,
            "t_speech_end_to_turn": (t_turn - speech_end) if speech_end else None,
        }
        self.metrics["turns"].append(m)
        self._set_state("thinking")
        self._turn_first_audio_at = None
        self._turn_audio_samples = 0
        self._turn_vision_dropped = False
        m["vision"] = self.vision_active
        with self._audio_lock:
            self._tts_stream_done = False
            self._turn_consumed_at_start = self._consumed_samples
        spoken_parts: t.List[str] = []
        tts: t.Optional[_WebsocketTTS] = None
        use_ws = self.cfg.tts_transport == "websocket"
        try:
            if use_ws:
                tts = _WebsocketTTS(self.cfg, self._append_avatar_pcm)
                open_task = asyncio.create_task(tts.open())
            else:
                open_task = None

            chunker = PhraseChunker(self.cfg.tts_min_chunk_chars)
            first_token_at: t.Optional[float] = None
            stream = await self._open_llm_stream(self._build_messages(user_text))
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                if first_token_at is None:
                    first_token_at = time.monotonic()
                    m["llm_ttft"] = first_token_at - t_turn
                    if self._turn_vision_dropped:
                        m["vision_dropped"] = True
                self._emit({"type": "avatar_text", "text": delta})
                for phrase in chunker.push(delta):
                    phrase = clean_for_speech(phrase)
                    if not phrase or not _SPEAKABLE.search(phrase):
                        continue
                    spoken_parts.append(phrase)
                    if use_ws:
                        if open_task is not None:
                            await open_task
                            open_task = None
                        assert tts is not None
                        await tts.send_text(phrase + " ")
                    else:
                        await self._http_tts(phrase)
            tail = clean_for_speech(chunker.flush())
            if tail and _SPEAKABLE.search(tail):
                spoken_parts.append(tail)
                if use_ws:
                    if open_task is not None:
                        await open_task
                        open_task = None
                    assert tts is not None
                    await tts.send_text(tail + " ")
                else:
                    await self._http_tts(tail)

            if use_ws:
                if open_task is not None:
                    await open_task
                    open_task = None
                assert tts is not None
                await tts.finish()
            fa = self._turn_first_audio_at
            if fa is not None:
                if first_token_at is not None:
                    m["tts_first_audio_after_first_token"] = fa - first_token_at
                m["tts_first_audio_after_turn"] = fa - t_turn
                if speech_end:
                    m["speech_end_to_first_audio"] = fa - speech_end
            m["tts_samples"] = self._turn_audio_samples
            if self._turn_audio_samples == 0 and spoken_parts:
                detail = (tts.error if tts is not None else None) or "empty stream"
                self._error(f"TTS produced no audio ({detail})")
            full = " ".join(spoken_parts).strip()
            m["avatar_text"] = full
            m["total"] = time.monotonic() - t_turn
            self._commit_history(user_text, full)
            self._pending_user_text = ""
        except asyncio.CancelledError:
            partial = " ".join(spoken_parts).strip()
            with self._audio_lock:
                heard = self._consumed_samples - self._turn_consumed_at_start
            m["cancelled"] = True
            m["avatar_text_partial"] = partial
            m["tts_samples"] = self._turn_audio_samples
            m["total"] = time.monotonic() - t_turn
            if partial and heard > 0:
                # the user actually heard part of it -> keep it in history
                self._commit_history(user_text, partial)
                self._pending_user_text = ""
            raise
        except Exception as exc:
            m["error"] = f"{type(exc).__name__}: {exc}"
            self._error(f"turn failed: {type(exc).__name__}: {exc}")
            self._commit_history(user_text, "")
        finally:
            with self._audio_lock:
                self._tts_stream_done = True
                empty = len(self._avatar_pcm) == 0
            if tts is not None:
                await tts.close()
            if empty and self._state in ("thinking", "speaking"):
                self._set_state("listening")

    # ---------------------------------------------------------------- #
    # history hand-off across sessions
    #
    # On ZeroGPU a conversation outlives its GPU lease: the worker is reclaimed
    # every SESSION_SECONDS and the next one is a fresh fork with a fresh brain.
    # Moving the plain [{role, content}] list across is what lets the avatar
    # keep talking about what it was just talking about. Only the LLM history
    # travels -- audio buffers, VAD state and the ElevenLabs socket are all
    # rebuilt, and the video rollout necessarily restarts.
    # ---------------------------------------------------------------- #
    def export_history(self) -> t.List[dict]:
        """A copy of the LLM turn history, safe to hand to a later brain."""
        return [dict(m) for m in self._history]

    def import_history(self, history) -> int:
        """Seed this brain's history. Returns how many messages were adopted.

        Defensive because the value round-trips through module state that
        outlives the session that produced it.
        """
        clean = []
        for m in history or []:
            if not isinstance(m, dict):
                continue
            role, content = m.get("role"), m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                clean.append({"role": role, "content": content})
        cap = self.cfg.history_max_messages * 2
        self._history = clean[-cap:]
        return len(self._history)

    def _commit_history(self, user_text: str, avatar_text: str) -> None:
        self._history.append({"role": "user", "content": user_text})
        if avatar_text:
            self._history.append({"role": "assistant", "content": avatar_text})
        if len(self._history) > self.cfg.history_max_messages * 2:
            self._history = self._history[-self.cfg.history_max_messages * 2:]

    async def _http_tts(self, text: str) -> None:
        """Fallback TTS: HTTP streaming endpoint, one request per phrase."""
        try:
            stream = self._eleven.text_to_speech.stream(  # type: ignore[union-attr]
                self.cfg.voice_id,
                text=text,
                model_id=self.cfg.tts_model,
                output_format="pcm_16000",
            )
            async for chunk in stream:
                if chunk:
                    self._append_avatar_pcm(chunk)
        except Exception as exc:
            self._error(f"HTTP TTS failed: {type(exc).__name__}: {exc}")


__all__ = ["ConversationBrain", "BrainConfig", "EnergyVAD", "PhraseChunker", "clean_for_speech"]

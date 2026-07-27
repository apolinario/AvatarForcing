"""Mock ConversationBrain — no ElevenLabs, no LLM, no network.

Implements EXACTLY the ``server/conversation.py`` ``ConversationBrain`` contract
from DESIGN.md::

    class ConversationBrain:
        def __init__(self, on_event, voice_id=None, system_prompt=None)
                                                          # on_event(dict), any thread
        async def start(self) -> None
        async def stop(self) -> None
        def feed_user_audio(self, pcm: np.int16[N]) -> None        # ~every 100 ms
        def set_user_snapshot(self, jpeg_bytes: bytes) -> None
        def pull_avatar_audio(self, n_samples: int) -> np.int16[n]  # silence when idle

Behaviour (a scripted stand-in for STT -> LLM -> TTS):

  listening: energy VAD over the fed PCM. Once ~1.5 s of *loud* audio has
             accumulated, partial ``user_transcript`` events are emitted; once
             ~0.6 s of *quiet* follows, the turn ends.
  thinking:  final ``user_transcript`` + ``{"state":"thinking"}``; 0.5 s later
  speaking:  ``avatar_text`` (streamed word by word) + ``{"state":"speaking"}``
             and ~2 s of clearly audible synthetic speech-ish audio (220-440 Hz
             chirp, syllable envelope) is queued for ``pull_avatar_audio``.
  listening: once the queued audio has been fully consumed.

Barge-in: loud user audio while speaking flushes the un-consumed buffer and
returns to listening, like the real brain.

Everything mutating shared state takes ``self._lock`` so the audio-feeding
thread, the WS thread and the asyncio loop can all touch it safely.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import numpy as np

__all__ = ["MockConversationBrain", "ConversationBrain"]

SR = 16000

# --- VAD / turn-taking tuning ------------------------------------------------
RMS_THRESHOLD = 0.02          # int16-normalised RMS above which audio is "loud"
MIN_SPEECH_SEC = 1.5          # loud audio needed to consider it a user turn
END_SILENCE_SEC = 0.6         # quiet needed after speech to close the turn
THINK_SEC = 0.5               # listening -> speaking latency
REPLY_SEC = 2.0               # length of the synthetic TTS reply
BARGE_IN_SEC = 0.35           # loud audio while speaking that triggers barge-in

FAKE_USER_LINES = [
    "Hey, can you see me right now?",
    "What do you think of my hat?",
    "Tell me something interesting.",
    "How is the latency feeling on your side?",
    "Okay, one more question for you.",
]
FAKE_AVATAR_LINES = [
    "Yes, I can see you clearly, and the framing looks great.",
    "That cap is a whole personality. I respect it.",
    "Here is something odd: we are talking through a mock pipeline right now.",
    "Feels smooth to me. Every block lands inside its four hundred milliseconds.",
    "Go ahead, I am listening.",
]


def _synth_reply(seconds: float = REPLY_SEC, sr: int = SR) -> np.ndarray:
    """A clearly audible, speech-ish chirp: 220 -> 440 Hz with syllables."""
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    f0 = 220.0 + 220.0 * (t / max(seconds, 1e-6))            # rising pitch
    f0 = f0 + 12.0 * np.sin(2 * np.pi * 4.5 * t)             # vibrato
    phase = 2 * np.pi * np.cumsum(f0) / sr
    tone = (
        0.62 * np.sin(phase)
        + 0.24 * np.sin(2 * phase)
        + 0.10 * np.sin(3 * phase)
    ).astype(np.float32)
    # syllable envelope (~4.5 syllables/s) + overall fade in/out
    syl = 0.35 + 0.65 * np.clip(np.sin(2 * np.pi * 4.5 * t) ** 2, 0.0, 1.0)
    fade = np.minimum(1.0, np.minimum(t / 0.04, (seconds - t) / 0.08)).clip(0.0, 1.0)
    wav = (tone * syl * fade * 0.34).astype(np.float32)
    return np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)


class MockConversationBrain:
    """Scripted, thread-safe stand-in for the real ConversationBrain."""

    def __init__(self, on_event, voice_id: str | None = None,
                 system_prompt: str | None = None) -> None:
        self._on_event = on_event
        # Per-session UI overrides (the real brain forwards them to
        # BrainConfig.voice_id / .system_prompt). The scripted replies ignore the
        # prompt; it is kept only so the handshake can be asserted end-to-end.
        # Instance lifetime only — never logged or emitted.
        self.voice_id = voice_id or None
        self.system_prompt = system_prompt or None
        self._lock = threading.RLock()

        self._state = "listening"
        self._buf = np.zeros(0, dtype=np.int16)      # pending avatar (TTS) audio
        self._snapshot: bytes | None = None
        self.n_snapshots = 0
        self.n_turns = 0

        # VAD accounting, in samples
        self._loud_samples = 0
        self._quiet_samples = 0
        self._speech_seen = False
        self._partials_emitted = 0
        self._turn_ready = False        # set by feed_user_audio, consumed by _run
        self._barge_loud = 0

        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._speak_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._reset_vad()
        self._task = asyncio.create_task(self._run(), name="mock-brain")
        with self._lock:
            self._state = "listening"
        self._emit({"type": "state", "value": "listening"})  # always announce once

    async def stop(self) -> None:
        self._running = False
        for t in (self._speak_task, self._task):
            if t is not None and not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
        self._speak_task = None
        self._task = None
        with self._lock:
            self._buf = np.zeros(0, dtype=np.int16)

    # ------------------------------------------------------------------ #
    # producer side (called from the WS / audio thread)
    # ------------------------------------------------------------------ #
    def feed_user_audio(self, pcm: np.ndarray) -> None:
        if pcm is None:
            return
        a = np.asarray(pcm)
        if a.size == 0:
            return
        if a.dtype == np.int16:
            f = a.astype(np.float32) / 32768.0
        else:
            f = a.astype(np.float32, copy=False)
            if np.max(np.abs(f), initial=0.0) > 1.5:  # looks like raw int16 in a float array
                f = f / 32768.0
        n = f.size
        rms = float(np.sqrt(np.mean(np.square(f), dtype=np.float64)))
        loud = rms >= RMS_THRESHOLD

        with self._lock:
            state = self._state
            if state == "speaking":
                # barge-in detection
                self._barge_loud = self._barge_loud + n if loud else 0
                if self._barge_loud >= BARGE_IN_SEC * SR:
                    self._barge_loud = 0
                    self._interrupt = True
                return
            if state == "thinking":
                return

            if loud:
                self._loud_samples += n
                self._quiet_samples = 0
                if self._loud_samples >= MIN_SPEECH_SEC * SR:
                    self._speech_seen = True
            else:
                if self._speech_seen:
                    self._quiet_samples += n
                    if self._quiet_samples >= END_SILENCE_SEC * SR:
                        self._turn_ready = True
                else:
                    # short blips decay so background noise never triggers a turn
                    self._loud_samples = max(0, self._loud_samples - n // 2)

    def set_user_snapshot(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._snapshot = jpeg_bytes
            self.n_snapshots += 1

    def pull_avatar_audio(self, n_samples: int) -> np.ndarray:
        n = int(n_samples)
        out = np.zeros(n, dtype=np.int16)
        with self._lock:
            take = min(n, self._buf.size)
            if take:
                out[:take] = self._buf[:take]
                self._buf = self._buf[take:]
        return out

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    _interrupt = False

    def _reset_vad(self) -> None:
        with self._lock:
            self._loud_samples = 0
            self._quiet_samples = 0
            self._speech_seen = False
            self._partials_emitted = 0
            self._turn_ready = False
            self._barge_loud = 0
            self._interrupt = False

    def _emit(self, ev: dict) -> None:
        try:
            self._on_event(ev)
        except Exception:  # never let a consumer bug kill the brain
            pass

    def _set_state(self, value: str) -> None:
        with self._lock:
            if self._state == value:
                return
            self._state = value
        self._emit({"type": "state", "value": value})

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    async def _run(self) -> None:
        """Poll the VAD flags at 50 ms and drive the scripted turn machine."""
        try:
            while self._running:
                await asyncio.sleep(0.05)
                with self._lock:
                    state = self._state
                    turn_ready = self._turn_ready
                    speech_seen = self._speech_seen
                    loud_s = self._loud_samples
                    partials = self._partials_emitted
                    interrupt = self._interrupt
                    buf_left = self._buf.size

                if state == "listening":
                    # stream a couple of fake partial transcripts while talking
                    if speech_seen and partials == 0:
                        with self._lock:
                            self._partials_emitted = 1
                        line = FAKE_USER_LINES[self.n_turns % len(FAKE_USER_LINES)]
                        self._emit({"type": "user_transcript",
                                    "text": " ".join(line.split()[:3]),
                                    "final": False})
                    elif turn_ready:
                        await self._do_turn()

                elif state == "speaking":
                    if interrupt:
                        with self._lock:
                            self._interrupt = False
                            self._buf = np.zeros(0, dtype=np.int16)
                        if self._speak_task and not self._speak_task.done():
                            self._speak_task.cancel()
                        self._reset_vad()
                        self._set_state("listening")
                    elif buf_left == 0 and (self._speak_task is None
                                            or self._speak_task.done()):
                        self._reset_vad()
                        self._set_state("listening")
        except asyncio.CancelledError:
            raise

    async def _do_turn(self) -> None:
        idx = self.n_turns % len(FAKE_USER_LINES)
        self.n_turns += 1
        user_line = FAKE_USER_LINES[idx]
        avatar_line = FAKE_AVATAR_LINES[idx]

        self._emit({"type": "user_transcript", "text": user_line, "final": True})
        self._set_state("thinking")
        with self._lock:
            self._turn_ready = False
            self._speech_seen = False
            self._loud_samples = 0
            self._quiet_samples = 0
        await asyncio.sleep(THINK_SEC)

        # queue the whole reply audio up front (real TTS streams it in chunks;
        # the web layer only ever sees pull_avatar_audio, so this is equivalent)
        audio = _synth_reply(REPLY_SEC)
        with self._lock:
            self._buf = np.concatenate([self._buf, audio])
        self._set_state("speaking")
        self._speak_task = asyncio.create_task(self._stream_text(avatar_line))

    async def _stream_text(self, line: str) -> None:
        """Emit avatar_text incrementally, roughly in sync with the audio."""
        words = line.split()
        per = max(0.05, (REPLY_SEC * 0.85) / max(1, len(words)))
        try:
            for i, w in enumerate(words):
                self._emit({"type": "avatar_text",
                            "text": (w if i == 0 else " " + w)})
                await asyncio.sleep(per)
        except asyncio.CancelledError:
            raise


ConversationBrain = MockConversationBrain


# --------------------------------------------------------------------------- #
# tiny self-test:  python -m server.mock_conversation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    async def main() -> None:
        events: list[dict] = []

        def on_event(ev: dict) -> None:
            events.append(ev)
            print(f"{time.strftime('%H:%M:%S')}  {ev}")

        brain = MockConversationBrain(on_event)
        await brain.start()

        def chunk(loud: bool, sec: float = 0.1) -> np.ndarray:
            n = int(sec * SR)
            if not loud:
                return (np.random.randn(n) * 30).astype(np.int16)
            t = np.arange(n) / SR
            return (np.sin(2 * np.pi * 180 * t) * 9000).astype(np.int16)

        loud_energy = 0
        t0 = time.time()
        while time.time() - t0 < 9.0:
            el = time.time() - t0
            brain.feed_user_audio(chunk(0.5 < el < 2.3))
            got = brain.pull_avatar_audio(1600)
            if brain.state == "speaking":
                loud_energy += int(np.abs(got).mean())
            await asyncio.sleep(0.1)

        print("states:", [e["value"] for e in events if e["type"] == "state"])
        print("mean |pcm| while speaking (sum of chunk means):", loud_energy)
        await brain.stop()

    asyncio.run(main())

"""Mock AvatarForcing streaming engine — CPU only, no torch, no cv2.

Implements EXACTLY the same interface/constants as ``server/engine.py``'s
``AvatarEngine`` / ``FaceCropper`` (see DESIGN.md) so that ``server/web.py`` can
be developed and smoke-tested without a GPU or model weights.

The rendered frames are deliberately "alive":
  * the reference photo is the base image (center-cropped + resized to 512),
  * a mouth-shaped ellipse opens/closes with the RMS of ``avatar_audio``
    (so TTS audio visibly animates the avatar),
  * an orbiting accent dot + rotating arc advance with the global frame index
    (so dropped / reordered / stalled frames are visible at a glance),
  * a HUD prints block index, frame-in-block, avatar RMS and user RMS,
  * a waveform strip of ``avatar_audio`` and a user-mic level bar are drawn.

Only numpy + Pillow are used.
"""

from __future__ import annotations

import io
import math
import os
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["MockAvatarEngine", "MockFaceCropper", "AvatarEngine", "FaceCropper"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def _rms(x: np.ndarray | None) -> float:
    if x is None or len(x) == 0:
        return 0.0
    a = np.asarray(x, dtype=np.float32)
    if a.dtype.kind in "iu":  # pragma: no cover - defensive
        a = a / 32768.0
    return float(np.sqrt(np.mean(np.square(a), dtype=np.float64)))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# FaceCropper mock
# --------------------------------------------------------------------------- #
class MockFaceCropper:
    """Passthrough stand-in for the face_alignment-based ``FaceCropper``.

    Contract (DESIGN.md)::

        def crop(self, frame_rgb: np.uint8[H, W, 3]) -> np.uint8[512, 512, 3]

    This mock simply center-crops to a square and resizes to 512x512, which is
    exactly what the real cropper degrades to when no face is detected.
    """

    SIZE = 512

    def __init__(self, size: int = 512, **_kwargs) -> None:
        self.SIZE = int(size)
        self._lock = threading.Lock()
        self.n_crops = 0

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame_rgb)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = _center_crop_square(Image.fromarray(arr[:, :, :3], "RGB"))
        if img.size != (self.SIZE, self.SIZE):
            img = img.resize((self.SIZE, self.SIZE), Image.BILINEAR)
        with self._lock:
            self.n_crops += 1
        return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# AvatarEngine mock
# --------------------------------------------------------------------------- #
class MockAvatarEngine:
    """Drop-in mock of ``server.engine.AvatarEngine``.

    Same constants, same method signatures, same array shapes/dtypes.
    ``step()`` sleeps ~80 ms to simulate GPU work (real budget: < 400 ms).
    """

    FPS = 25
    SR = 16000
    BLOCK_FRAMES = 10
    BLOCK_SAMPLES = 6400
    PRIME_FRAMES = 50
    SIZE = 512

    #: simulated GPU latency per step, seconds (override for benchmarking)
    STEP_LATENCY = 0.080

    def __init__(
        self,
        ref_image_path: str,
        repo_dir: str = "AvatarForcing",
        device: str = "cuda",
    ) -> None:
        self.ref_image_path = ref_image_path
        self.repo_dir = repo_dir
        self.device = device

        self._loaded = False
        self.reference_is_default = True
        self._base: Image.Image | None = None
        self._font_hud = _font(15)
        self._font_big = _font(20)

        # session state
        self._lock = threading.Lock()
        self.block_idx = 0          # next block index that step() will render
        self.global_frame = 0       # frames emitted since start_session (excl. priming)
        self.session_id = 0
        self.n_steps = 0
        self.total_step_time = 0.0

        # mouth geometry in 512-space (tuned for photo_2.jpeg)
        self._mouth_cx = 0.685
        self._mouth_cy = 0.767
        self._mouth_hw = 30

    # ---------------- lifecycle ---------------- #
    def load(self) -> None:
        """Cheap stand-in for loading checkpoints + precomputing ref latents."""
        if self._loaded:
            return
        img: Image.Image
        try:
            img = Image.open(self.ref_image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.SIZE, self.SIZE), (26, 28, 34))
            d = ImageDraw.Draw(img)
            d.text((16, 16), "no ref image", fill=(220, 90, 90), font=self._font_big)
        img = _center_crop_square(img)
        if img.size != (self.SIZE, self.SIZE):
            img = img.resize((self.SIZE, self.SIZE), Image.LANCZOS)
        self._base = img
        # a slightly darkened copy used to paint the "open mouth" cavity
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def set_reference(self, image=None) -> float:
        """Mirror of ``AvatarEngine.set_reference``: swap the base identity.

        ``image``: encoded image bytes, an RGB array, or ``None`` for the
        process default (``ref_image_path``). The real one costs ~56 ms on the
        GPU thread; this one is a decode + resize.
        """
        t0 = time.perf_counter()
        self._ensure_loaded()
        if image is None:
            src: object = self.ref_image_path
        elif isinstance(image, np.ndarray):
            src = Image.fromarray(np.asarray(image, dtype=np.uint8)[:, :, :3], "RGB")
        else:
            src = io.BytesIO(bytes(image))
        img = src if isinstance(src, Image.Image) else Image.open(src)  # type: ignore[arg-type]
        img = _center_crop_square(img.convert("RGB"))
        if img.size != (self.SIZE, self.SIZE):
            img = img.resize((self.SIZE, self.SIZE), Image.LANCZOS)
        self._base = img
        self.reference_is_default = image is None
        return time.perf_counter() - t0

    def start_session(self, first_user_frame: np.ndarray | None) -> np.ndarray:
        """Reset state and return the 50 primed RGB frames (2 s @ 25 fps)."""
        self._ensure_loaded()
        time.sleep(0.25)  # simulate KV-cache reset + priming block on GPU
        with self._lock:
            self.block_idx = 0
            self.global_frame = 0
            self.session_id += 1
            self.n_steps = 0
            self.total_step_time = 0.0
        out = np.empty((self.PRIME_FRAMES, self.SIZE, self.SIZE, 3), dtype=np.uint8)
        for i in range(self.PRIME_FRAMES):
            out[i] = self._render(
                global_frame=i - self.PRIME_FRAMES,
                block_idx=-1,
                frame_in_block=i,
                a_level=0.0,
                u_level=0.0,
                wave=None,
                label="priming",
            )
        return out

    def step(
        self,
        avatar_audio: np.ndarray,
        user_audio: np.ndarray,
        user_frames: np.ndarray,
    ) -> np.ndarray:
        """Render one 400 ms block of 10 frames. Simulates ~80 ms of GPU work."""
        self._ensure_loaded()
        t0 = time.perf_counter()

        avatar_audio = self._as_audio(avatar_audio, "avatar_audio")
        user_audio = self._as_audio(user_audio, "user_audio")
        if user_frames is not None:
            uf = np.asarray(user_frames)
            if uf.ndim != 4 or uf.shape[0] != self.BLOCK_FRAMES:
                raise ValueError(
                    f"user_frames must be [{self.BLOCK_FRAMES},{self.SIZE},"
                    f"{self.SIZE},3], got {getattr(uf, 'shape', None)}"
                )

        with self._lock:
            b = self.block_idx
            g0 = self.global_frame
            self.block_idx += 1
            self.global_frame += self.BLOCK_FRAMES

        a_rms = _rms(avatar_audio)
        u_rms = _rms(user_audio)

        # per-frame envelope of the avatar audio -> mouth opening
        seg = np.asarray(avatar_audio, dtype=np.float32)
        chunks = np.array_split(seg, self.BLOCK_FRAMES)
        levels = [float(np.sqrt(np.mean(np.square(c)))) if len(c) else 0.0 for c in chunks]

        # coarse waveform for the HUD strip
        wave = np.abs(seg).reshape(-1)
        step = max(1, len(wave) // 96)
        wave = wave[: step * 96].reshape(96, step).max(axis=1) if len(wave) >= 96 else None

        # simulate GPU compute (denoise + decode) before producing pixels
        elapsed = time.perf_counter() - t0
        if self.STEP_LATENCY > elapsed:
            time.sleep(self.STEP_LATENCY - elapsed)

        out = np.empty((self.BLOCK_FRAMES, self.SIZE, self.SIZE, 3), dtype=np.uint8)
        for f in range(self.BLOCK_FRAMES):
            out[f] = self._render(
                global_frame=g0 + f,
                block_idx=b,
                frame_in_block=f,
                a_level=levels[f],
                u_level=u_rms,
                wave=wave,
                label=None,
            )

        dt = time.perf_counter() - t0
        with self._lock:
            self.n_steps += 1
            self.total_step_time += dt
        return out

    # ---------------- introspection (mock-only, harmless) ---------------- #
    @property
    def avg_step_ms(self) -> float:
        with self._lock:
            return 0.0 if not self.n_steps else 1000.0 * self.total_step_time / self.n_steps

    def close(self) -> None:  # symmetry with a possible real engine
        pass

    # ---------------- internals ---------------- #
    def _as_audio(self, x: np.ndarray, name: str) -> np.ndarray:
        if x is None:
            return np.zeros(self.BLOCK_SAMPLES, dtype=np.float32)
        a = np.asarray(x)
        if a.dtype.kind in "iu":
            a = a.astype(np.float32) / 32768.0
        else:
            a = a.astype(np.float32, copy=False)
        a = a.reshape(-1)
        if len(a) != self.BLOCK_SAMPLES:
            fixed = np.zeros(self.BLOCK_SAMPLES, dtype=np.float32)
            n = min(len(a), self.BLOCK_SAMPLES)
            fixed[self.BLOCK_SAMPLES - n:] = a[-n:]
            a = fixed
        return a

    def _render(
        self,
        global_frame: int,
        block_idx: int,
        frame_in_block: int,
        a_level: float,
        u_level: float,
        wave: np.ndarray | None,
        label: str | None,
    ) -> np.ndarray:
        assert self._base is not None
        im = self._base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        S = self.SIZE

        # ---- mouth: opens with avatar audio level ----
        openness = min(1.0, a_level * 7.0)
        cx, cy = self._mouth_cx * S, self._mouth_cy * S
        hw = self._mouth_hw * (1.0 + 0.10 * openness)
        hh = 2.5 + 20.0 * openness
        d.ellipse(
            (cx - hw, cy - hh, cx + hw, cy + hh),
            fill=(58, 22, 26, 235),
            outline=(120, 60, 62, 180),
            width=2,
        )
        if openness > 0.15:  # teeth + tongue hint so the motion reads clearly
            d.ellipse(
                (cx - hw * 0.8, cy - hh, cx + hw * 0.8, cy - hh + max(3, hh * 0.35)),
                fill=(226, 220, 214, 220),
            )
            d.ellipse(
                (cx - hw * 0.45, cy + hh * 0.15, cx + hw * 0.45, cy + hh * 0.9),
                fill=(150, 60, 70, 200),
            )
        # subtle "breathing" brightness so idle frames are not identical
        glow = int(10 * (0.5 + 0.5 * math.sin(global_frame * 0.12)))
        if glow:
            d.rectangle((0, 0, S, S), fill=(255, 245, 230, glow))

        # ---- moving element: orbiting dot + rotating arc (frame-clock proof) ----
        ang = global_frame * 0.20
        ox = S * 0.5 + math.cos(ang) * S * 0.44
        oy = S * 0.5 + math.sin(ang) * S * 0.44
        d.ellipse((ox - 9, oy - 9, ox + 9, oy + 9), fill=(120, 230, 200, 235))
        d.arc((14, 14, 74, 74), start=(global_frame * 9) % 360,
              end=(global_frame * 9 + 110) % 360, fill=(120, 230, 200, 255), width=5)

        # ---- HUD ----
        d.rectangle((0, S - 58, S, S), fill=(8, 10, 14, 165))
        if wave is not None and len(wave):
            n = len(wave)
            w = (S - 20) / n
            for i, v in enumerate(wave):
                h = min(24.0, float(v) * 42.0)
                x = 10 + i * w
                d.rectangle((x, S - 30 - h, x + max(1.0, w - 1), S - 30),
                            fill=(120, 230, 200, 230))
        # user mic level bar
        d.rectangle((10, S - 22, 10 + (S - 20) * min(1.0, u_level * 6.0), S - 14),
                    fill=(240, 170, 90, 235))

        d.rectangle((0, 0, S, 26), fill=(8, 10, 14, 160))
        txt = (
            f"MOCK  blk {block_idx:>4}  f {frame_in_block}  g {global_frame:>6}  "
            f"a {a_level:0.3f}  u {u_level:0.3f}"
        )
        if label:
            txt = f"MOCK  {label}  f {frame_in_block}"
        d.text((8, 5), txt, fill=(226, 232, 240, 255), font=self._font_hud)

        return np.asarray(im, dtype=np.uint8)


# Aliases so `from server.mock_engine import AvatarEngine, FaceCropper` mirrors
# `from server.engine import AvatarEngine, FaceCropper`.
AvatarEngine = MockAvatarEngine
FaceCropper = MockFaceCropper


def default_ref_image() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "photo_2.jpeg")

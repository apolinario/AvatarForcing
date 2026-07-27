"""Real-time streaming engine for AvatarForcing (Agent C).

This module re-formulates ``AvatarForcing.sample()`` (which is written as an
offline, whole-utterance block loop) into an **incremental** engine that
produces one 10-frame / 400 ms block per :meth:`AvatarEngine.step` call and can
run indefinitely.

Public contract (see ``DESIGN.md``)::

    class AvatarEngine:
        FPS = 25; SR = 16000
        BLOCK_FRAMES = 10; BLOCK_SAMPLES = 6400
        PRIME_FRAMES = 50; SIZE = 512

        def __init__(self, ref_image_path, repo_dir=<repo root>, device="cuda")
        def load(self) -> None
        def set_reference(image=None) -> float          # None -> ref_image_path
        def start_session(first_user_frame: uint8[H,W,3] | None) -> uint8[50,512,512,3]
        def step(avatar_audio: f32[6400], user_audio: f32[6400],
                 user_frames: uint8[10,512,512,3]) -> uint8[10,512,512,3]

    class FaceCropper:
        def crop(frame_rgb: uint8[H,W,3]) -> uint8[512,512,3]

Mapping onto ``sample()``
-------------------------
``sample()`` does, for a T-frame utterance:

1. encode the *entire* avatar/user waveforms with wav2vec2 -> ``[1, T, 512]``,
   encode *all* user frames -> ``[1, T, 512]``;
2. block 0 (frames ``[0, 50)``): fresh noise, ``prepare_cfg_condition(seq_len=50,
   context_len=0)``, ``nfe-1`` ``solve_cfg`` steps with ``use_kv_cache=False``,
   ``start_pos=0``, then one ``update_kv_cache``;
3. blocks ``t = 50, 60, ...``: ``x_t = cat(last 2 clean latents, randn(10))``,
   conditions sliced ``[t-2, t+10)``, ``prepare_cfg_condition(seq_len=10,
   context_len=2)``, ``nfe-1`` ``solve_cfg`` steps with ``use_kv_cache=True`` and
   ``start_pos = t-2``, then one ``update_kv_cache``.

:meth:`AvatarEngine.start_session` is step (2) with 2 s of silence + the first
user frame repeated 50x. :meth:`AvatarEngine.step` is exactly one iteration of
step (3); the only change is that the conditions come from *rolling* buffers
instead of pre-computed full-utterance tensors. Every tensor-level call
(``prepare_cfg_condition`` / ``solve_cfg`` / ``update_kv_cache`` /
``decode_block``) is the upstream implementation, unmodified.

See ``reports/engine.md`` for the audio-window design, the rotary-embedding fix
that lifts the ~41 s session limit, validation results and benchmarks.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Optional, Sequence

import numpy as np

__all__ = ["AvatarEngine", "FaceCropper"]


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_THIS_DIR)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _resolve_repo(repo_dir: str) -> str:
    if os.path.isabs(repo_dir):
        return repo_dir
    return os.path.join(_APP_DIR, repo_dir)


# --------------------------------------------------------------------------- #
# shared SFD face detector (read-only after construction -> safe to share)
# --------------------------------------------------------------------------- #
_DETECTOR_LOCK = threading.Lock()
_DETECTORS: dict = {}


def _cudnn_flags():
    import torch

    return (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)


def _set_cudnn_flags(flags) -> None:
    import torch

    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags


def get_shared_face_detector(device: str = "cuda"):
    """Lazily build (and cache per device) a bare SFD face detector.

    ``face_alignment.FaceAlignment`` is deliberately *not* used:

    * it also loads / ``torch.compile``s the 2DFAN landmark network (~40 s) that
      AvatarForcing never uses, and
    * its ``__init__`` flips ``torch.backends.cudnn.benchmark = True``, which
      turns the first pass through the motion encoder into a 14-27 s cudnn
      autotune (measured by Agent A).

    ``face_alignment.detection.sfd.detect.batch_detect`` *also* sets
    ``cudnn.benchmark = True`` on every call, so every detection in this module
    is wrapped in a save/restore of the cudnn flags.
    """
    key = str(device)
    with _DETECTOR_LOCK:
        det = _DETECTORS.get(key)
        if det is None:
            flags = _cudnn_flags()
            try:
                from face_alignment.detection.sfd import FaceDetector as SFDDetector

                det = SFDDetector(device=key, verbose=False)
            finally:
                _set_cudnn_flags(flags)
            _DETECTORS[key] = det
        return det


# --------------------------------------------------------------------------- #
# audio level tracking / normalisation
# --------------------------------------------------------------------------- #
class _AudioNormalizer:
    """Streaming stand-in for ``Wav2Vec2FeatureExtractor``'s utterance-level
    zero-mean/unit-variance normalisation.

    The offline pipeline computes ``(x - x.mean()) / sqrt(x.var() + 1e-7)`` over
    the **whole** utterance. Doing that per rolling window is wrong: a 2.4 s
    window of near-silence has ``std ~= 4e-4`` while ``data/avatar.wav``'s global
    std is ``1.7e-2``, so a silent window would be amplified ~40x and the avatar
    would babble during pauses.

    Instead we track a *speech level* (EMA of the RMS of voiced 400 ms chunks,
    gated against a decaying peak) and divide every window by it. On
    ``data/avatar.wav`` that estimator settles at ~1.7e-2, i.e. essentially the
    offline global std, while silence stays silent.
    """

    def __init__(
        self,
        fixed_std: Optional[float] = None,
        init_std: float = 0.02,
        floor: float = 0.005,
        ceil: float = 0.15,
        ema: float = 0.08,
        peak_decay: float = 0.98,
        voiced_frac: float = 0.25,
        abs_floor: float = 1e-3,
    ) -> None:
        self.fixed_std = fixed_std
        self.init_std = float(init_std)
        self.floor = float(floor)
        self.ceil = float(ceil)
        self.ema = float(ema)
        self.peak_decay = float(peak_decay)
        self.voiced_frac = float(voiced_frac)
        self.abs_floor = float(abs_floor)
        self.reset()

    def reset(self) -> None:
        self.level = self.init_std
        self.peak = 0.0
        self.mean = 0.0
        self.n_voiced = 0

    def update(self, chunk: np.ndarray) -> None:
        if self.fixed_std is not None or chunk.size == 0:
            return
        x = chunk.astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(x * x)))
        self.mean = 0.98 * self.mean + 0.02 * float(x.mean())
        self.peak = max(rms, self.peak * self.peak_decay)
        voiced = rms > max(self.abs_floor, self.voiced_frac * self.peak)
        if voiced:
            if self.n_voiced == 0:
                self.level = rms
            else:
                self.level = (1.0 - self.ema) * self.level + self.ema * rms
            self.n_voiced += 1
        self.level = min(max(self.level, self.floor), self.ceil)

    @property
    def std(self) -> float:
        return float(self.fixed_std) if self.fixed_std is not None else float(self.level)

    def normalize(self, window: np.ndarray) -> np.ndarray:
        std = self.std
        mean = 0.0 if self.fixed_std is not None else self.mean
        return ((window.astype(np.float32) - np.float32(mean)) / np.float32(std)).astype(np.float32)


# --------------------------------------------------------------------------- #
# geometry helpers (mirror DataProcessor.preprocess_face)
# --------------------------------------------------------------------------- #
def _square_box(mx: float, my: float, half: float, w: int, h: int):
    """``preprocess_face``'s clamp-then-re-square dance, verbatim."""
    bs = int(half)
    x1, y1 = int(mx) - bs, int(my) - bs
    x2, y2 = int(mx) + bs, int(my) + bs
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    bsx, bsy = x2 - x1, y2 - y1
    cx, cy = int(x1 + bsx // 2), int(y1 + bsy // 2)
    bs = int(min(bsx, bsy) // 2)
    bs = max(bs, 1)
    x1, y1 = max(cx - bs, 0), max(cy - bs, 0)
    return x1, y1, x1 + 2 * bs, y1 + 2 * bs


class FaceCropper:
    """face_alignment(SFD)-based square face cropper for webcam frames.

    Detection is expensive (~15-25 ms), so it runs only every
    ``detect_every`` crops; in between, the last (EMA-smoothed) box is reused,
    which makes :meth:`crop` a pure slice + ``cv2.resize`` (<1 ms). When no
    confident face is found we keep the previous box, or fall back to the centre
    square, so a user turning away can never kill the session.
    """

    SIZE = 512

    def __init__(
        self,
        size: int = 512,
        device: str = "cuda",
        detect_every: int = 50,
        retry_every: int = 10,
        pad_ratio: float = 1.0,
        score_thresh: float = 0.95,
        ema: float = 0.25,
        detector=None,
        detect_height: float = 360.0,
    ) -> None:
        self.SIZE = int(size)
        self.device = device
        self.detect_every = int(detect_every)
        self.retry_every = int(retry_every)
        self.pad_ratio = float(pad_ratio)
        self.score_thresh = float(score_thresh)
        self.ema = float(ema)
        self.detect_height = float(detect_height)
        self._detector = detector
        # smoothed box state (in full-frame pixels)
        self._mx: Optional[float] = None
        self._my: Optional[float] = None
        self._half: Optional[float] = None
        self.n_crops = 0
        self.n_detect = 0
        self.n_detect_fail = 0
        self.last_detect_ms = 0.0

    # -------------------------------------------------- #
    @property
    def detector(self):
        if self._detector is None:
            self._detector = get_shared_face_detector(self.device)
        return self._detector

    def detect_box(self, frame_rgb: np.ndarray):
        """Return ``(mx, my, half)`` of the padded square box, or ``None``."""
        import cv2

        h, w = frame_rgb.shape[:2]
        mult = self.detect_height / float(h)
        if abs(mult - 1.0) > 1e-3:
            interp = cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC
            small = cv2.resize(frame_rgb, dsize=(0, 0), fx=mult, fy=mult, interpolation=interp)
        else:
            mult, small = 1.0, frame_rgb

        t0 = time.perf_counter()
        flags = _cudnn_flags()
        try:
            boxes = self.detector.detect_from_image(np.ascontiguousarray(small))
        except Exception:
            boxes = []
        finally:
            _set_cudnn_flags(flags)
        self.last_detect_ms = (time.perf_counter() - t0) * 1e3
        self.n_detect += 1

        boxes = [b for b in boxes if float(b[4]) > self.score_thresh]
        if not boxes:
            self.n_detect_fail += 1
            return None
        x1, y1, x2, y2 = (float(v) / mult for v in boxes[0][:4])
        bsy = (y2 - y1) / 2.0
        bsx = (x2 - x1) / 2.0
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(bsy, bsx) * (1.0 + self.pad_ratio)
        return mx, my, half

    def _update_box(self, box) -> None:
        mx, my, half = box
        if self._mx is None:
            self._mx, self._my, self._half = mx, my, half
        else:
            a = self.ema
            self._mx = (1 - a) * self._mx + a * mx
            self._my = (1 - a) * self._my + a * my
            self._half = (1 - a) * self._half + a * half

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2

        arr = np.asarray(frame_rgb)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = arr[:, :, :3]
        h, w = arr.shape[:2]

        n = self.n_crops
        self.n_crops += 1
        need = (self._half is None and n % max(1, self.retry_every) == 0) or (
            self._half is not None and n % max(1, self.detect_every) == 0
        )
        if need:
            box = self.detect_box(arr)
            if box is not None:
                self._update_box(box)

        if self._half is None:  # never saw a face -> centre square
            s = min(h, w)
            x1, y1 = (w - s) // 2, (h - s) // 2
            x2, y2 = x1 + s, y1 + s
        else:
            x1, y1, x2, y2 = _square_box(self._mx, self._my, self._half, w, h)

        face = arr[y1:y2, x1:x2]
        if face.size == 0:
            s = min(h, w)
            face = arr[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
        if face.shape[0] != self.SIZE or face.shape[1] != self.SIZE:
            interp = cv2.INTER_AREA if face.shape[0] > self.SIZE else cv2.INTER_CUBIC
            face = cv2.resize(face, (self.SIZE, self.SIZE), interpolation=interp)
        return np.ascontiguousarray(face, dtype=np.uint8)

    # convenience for warmup / bench
    def reset(self) -> None:
        self._mx = self._my = self._half = None
        self.n_crops = 0


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
class AvatarEngine:
    FPS = 25
    SR = 16000
    BLOCK_FRAMES = 10
    BLOCK_SAMPLES = 6400
    PRIME_FRAMES = 50
    SIZE = 512

    #: audio window (in 25 fps feature frames) re-encoded by wav2vec2 each step.
    #: 60 frames = 2.4 s = 50 frames of left context + the 10 new frames, i.e.
    #: exactly ``PRIME_FRAMES * 640 + BLOCK_SAMPLES`` raw samples.
    AUDIO_WINDOW_FRAMES = 60

    def __init__(
        self,
        ref_image_path: str,
        repo_dir: str = _APP_DIR,   # this repo root; server/ lives inside it
        device: str = "cuda",
        *,
        nfe: int = 10,
        a_cfg_scale: float = 2.0,
        u_cfg_scale: float = 1.0,
        seed: Optional[int] = 25,
        audio_window_frames: Optional[int] = None,
        avatar_norm_std: Optional[float] = None,
        user_norm_std: Optional[float] = None,
        rope_max_seq_len: int = 1 << 16,
        rope_rebase_margin: int = 64,
        warmup_steps: int = 4,
        detector=None,
        pose_gain: Optional[float] = None,
        pose_deadband: Optional[float] = None,
        pose_ema: Optional[float] = None,
        pose_max_step: Optional[float] = None,
    ) -> None:
        self.ref_image_path = ref_image_path
        self.repo_dir = _resolve_repo(repo_dir)
        self.device = device
        self.nfe = int(nfe)
        self.a_cfg_scale = float(a_cfg_scale)
        self.u_cfg_scale = float(u_cfg_scale)
        self.seed = seed
        self.audio_window_frames = int(audio_window_frames or self.AUDIO_WINDOW_FRAMES)
        self.rope_max_seq_len = int(rope_max_seq_len)
        self.rope_rebase_margin = int(rope_rebase_margin)
        self.warmup_steps = int(warmup_steps)
        self._detector = detector

        # --- pose anchor (drift controller, see _anchor_pose) --- #
        # gain 0 disables it and restores the plain upstream rollout. Deadband is
        # a fraction of |r_s| so it transfers across reference identities: healthy
        # rollouts measured at 0.15-0.18 |r_s|, so 0.30 leaves 2x headroom for
        # genuine pose changes before the controller does anything at all.
        self.pose_gain = _envf("AVATAR_POSE_GAIN", 0.25) if pose_gain is None else float(pose_gain)
        self.pose_deadband = (
            _envf("AVATAR_POSE_DEADBAND", 0.25) if pose_deadband is None else float(pose_deadband)
        )
        # EMA over block means; 0.08 ~ 5 s time constant, i.e. slower than any
        # expression or syllable, so only the pose term is ever measured.
        self.pose_ema = _envf("AVATAR_POSE_EMA", 0.08) if pose_ema is None else float(pose_ema)
        # Slew cap, also as a fraction of |r_s|: this is what bounds visibility,
        # so the gain can be set for authority rather than for subtlety. Measured
        # natural frame-to-frame motion is |dz| ~ 0.4 (p90 0.7); 0.05 |r_s| = 1.4
        # per block spread over 10 ramped frames is 0.14 per frame, i.e. a third
        # of the median natural step, and a smooth DC ramp rather than jitter.
        self.pose_max_step = (
            _envf("AVATAR_POSE_MAX_STEP", 0.05) if pose_max_step is None else float(pose_max_step)
        )
        self._pose_mu = None
        self._pose_deadband = 0.0
        self.pose_dist = 0.0
        self.n_pose_corr = 0

        self._loaded = False
        self._weights_loaded = False
        #: False once set_reference() has swapped in a user-supplied identity.
        #: The web layer reads it to avoid re-running the (GPU-bound) default
        #: recompute for sessions that never uploaded anything.
        self.reference_is_default = True
        self.G = None
        self.opt = None
        self.session_id = 0
        self.n_steps = 0
        self.n_rope_rebase = 0
        self._t = 0  # index of the next latent frame to generate

        self._a_norm = _AudioNormalizer(fixed_std=avatar_norm_std)
        self._u_norm = _AudioNormalizer(fixed_std=user_norm_std)

        self._audio_window_samples = self.audio_window_frames * (self.SR // self.FPS)
        self._a_ring = np.zeros(0, dtype=np.float32)
        self._u_ring = np.zeros(0, dtype=np.float32)
        self._gen = None
        self._rope_base = 0
        self._rope_len = self.rope_max_seq_len

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Weights + warm-up, the dedicated-GPU path.

        ZeroGPU splits these two halves across the process boundary — see
        :meth:`load_weights` and :meth:`warm`.
        """
        self.load_weights()
        self.warm()

    def load_weights(self) -> None:
        """Build the graph and get every parameter onto ``self.device``.

        Runs in the **parent** process on ZeroGPU (``app.py`` calls it at import).
        Nothing here executes a forward pass, which is what makes that legal:
        ZeroGPU's function mode intercepts ``.to("cuda")`` / ``copy_`` and keeps
        the real storage on CPU behind a fake-CUDA alias, then packs it so the
        first ``@spaces.GPU`` entry restores it straight into VRAM. A forward
        pass here would silently run on CPU and take minutes, hence :meth:`warm`.
        """
        if self._weights_loaded:
            return
        import torch
        from omegaconf import OmegaConf

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        cfg_path = os.path.join(self.repo_dir, "configs", "inference.yaml")
        opt = OmegaConf.load(cfg_path)
        opt = OmegaConf.merge(
            opt,
            OmegaConf.create(
                {
                    "rank": 0 if self.device.startswith("cuda") else self.device,
                    "ngpus": 1,
                    "nfe": self.nfe,
                    "a_cfg_scale": self.a_cfg_scale,
                    "u_cfg_scale": self.u_cfg_scale,
                    "wav2vec_model_path": os.path.join(self.repo_dir, opt.wav2vec_model_path),
                    "pretrained_dir": os.path.join(self.repo_dir, opt.pretrained_dir),
                }
            ),
        )
        self.opt = opt
        self.torch = torch

        from models.avatarforcing.AvatarForcing import AvatarForcing

        # cudnn: deterministic + no autotune (see get_shared_face_detector docstring)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        G = AvatarForcing(opt).to(self.device)
        with torch.no_grad():
            for ckpt in (
                os.path.join(self.repo_dir, "pretrained_dir", "motion_autoencoder.pth"),
                os.path.join(self.repo_dir, "pretrained_dir", "flow_transformer.pth"),
            ):
                # map_location="cpu" (not self.device): under ZeroGPU's parent
                # patching a "cuda" map_location would route the whole 800 MB
                # through the storage interception path for no gain — the
                # copy_ below already lands each tensor on the device.
                sd = torch.load(ckpt, map_location="cpu", weights_only=True)
                for name, param in G.named_parameters():
                    if name in sd:
                        param.copy_(sd[name].to(self.device))
                del sd
        G.eval()
        self.G = G

        # nfe schedule (sample() sets this per call; we set it once)
        G.denoising_step_list = torch.tensor(
            np.linspace(opt.num_train_timestep, 0, self.nfe - 1).tolist()
        )

        # --- trap (a): rotary table is precomputed for max_seq_len=1024 only --- #
        self._install_big_rope_table()

        self._weights_loaded = True

    def warm(self) -> None:
        """Reference latents + cuDNN/cuBLAS warm-up. Needs a **real** GPU.

        On ZeroGPU this runs inside the held ``@spaces.GPU`` lease (see
        ``server/gpu_session.py``), right after the packed weights are restored
        to VRAM and before ``__READY__`` is announced.
        """
        if self._loaded:
            return
        self.load_weights()
        import torch

        # --- reference-image latents (once per engine) --- #
        self._precompute_reference()

        # --- warm up everything (cudnn algo selection, lazy inits, cublas) --- #
        self._warmup()

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._loaded = True

    # ------------------------------------------------------------------ #
    def _install_big_rope_table(self) -> None:
        """Enlarge the rotary ``freqs_cis`` table (trap (a) from Agent A).

        ``Attention`` registers ``freqs_cis = precompute_freqs_cis(head_dim,
        max_seq_len=1024)``. In KV-cache mode ``start_pos = t - 2`` grows without
        bound, and ``apply_rotary_emb_q`` slices ``freqs_cis[start_pos:start_pos+12]``
        then ``.view(1, 1, 12, -1)`` -> once ``start_pos + 12 > 1024`` the view
        throws, i.e. the session dies after ~1022 latent frames (~41 s).

        Fix: build **one** table with ``rope_max_seq_len`` positions and share it
        across all 8 layers (33 MB for 65536 positions -> 43 min of continuous
        conversation). Beyond that :meth:`_maybe_rebase_rope` phase-rotates the
        cached keys back, which makes the session length unbounded.
        """
        import torch
        from models.avatarforcing.flow_transformer import precompute_freqs_cis

        attn0 = self.G.flow_transformer.blocks[0].attn
        head_dim = attn0.head_dim
        table = precompute_freqs_cis(head_dim, self.rope_max_seq_len).to(self.device)
        for blk in self.G.flow_transformer.blocks:
            blk.attn.freqs_cis = table
        self._rope_table = table
        self._rope_len = int(table.shape[0])
        self._rope_base = 0

    def _maybe_rebase_rope(self, start_pos: int) -> int:
        """Keep ``start_pos`` inside the rotary table, exactly.

        RoPE is a per-position *phase* multiplication, and in KV-cache mode
        ``start_pos`` only ever matters through the *relative* offsets between the
        38 cached keys and the 12 new tokens (both attention masks are static).
        A cached key stored at absolute position ``p`` can therefore be moved to
        ``p - delta`` by multiplying it with ``conj(freqs_cis[delta])`` — phases
        add, so this is exact. That lets us subtract ``delta`` from ``start_pos``
        whenever it approaches the end of the table.
        """
        import torch

        margin = self.rope_rebase_margin
        if start_pos + self.BLOCK_FRAMES + 2 <= self._rope_len - margin:
            return start_pos

        target = 64 + (start_pos % self.BLOCK_FRAMES)  # keep phase alignment tidy
        delta = start_pos - target
        if delta <= 0:
            raise RuntimeError("rope table too small to rebase")
        phase = self._rope_table[delta].conj()
        for self_kv, _cross_kv in self.G.kv_cache:
            k = self_kv["k"]
            kc = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2).contiguous())
            kc = kc * phase
            self_kv["k"] = torch.view_as_real(kc).flatten(-2).to(k.dtype).contiguous()
        self._rope_base += delta
        self.n_rope_rebase += 1
        return start_pos - delta

    # ------------------------------------------------------------------ #
    def _precompute_reference(self, source=None) -> None:
        import torch

        face = self.preprocess_reference(self.ref_image_path if source is None else source)
        self.ref_face = face
        s = self._frames_to_tensor(face[None])  # [1,3,512,512]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            s_r, r_s_lambda, s_r_feats = self.G.encode_image_into_latent(s)
            r_s = self.G.motion_autoencoder.dec.direction(r_s_lambda)
        self.s_r = s_r
        self.r_s = r_s
        self.s_r_feats = s_r_feats
        # pre-expanded for decode_block (10 frames per block)
        self._s_r_dec = s_r.unsqueeze(1)
        self._s_r_feats_exp = [f.repeat_interleave(self.BLOCK_FRAMES, dim=0) for f in s_r_feats]

        # pose-anchor constants: setpoint, absolute deadband and the in-block ramp
        self._r_s_row = r_s.reshape(1, -1).float()
        _rn = float(self._r_s_row.norm())
        self._pose_deadband = self.pose_deadband * _rn
        self._pose_max_step = self.pose_max_step * _rn
        self._pose_ramp = (
            torch.arange(1, self.BLOCK_FRAMES + 1, device=self.device, dtype=torch.float32)
            / self.BLOCK_FRAMES
        ).reshape(1, self.BLOCK_FRAMES, 1)

    def preprocess_reference(self, image) -> np.ndarray:
        """``DataProcessor.preprocess_face`` with a graceful no-face fallback.

        ``image`` is a file path, encoded image bytes (whatever ``cv2.imdecode``
        reads: JPEG/PNG/WebP) or an RGB uint8 array — the bytes form is what a
        user upload arrives as.
        """
        import cv2

        if isinstance(image, (bytes, bytearray, memoryview)):
            img = cv2.imdecode(np.frombuffer(bytes(image), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"could not decode reference image ({len(image)} bytes)")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
        else:
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropper = FaceCropper(size=self.SIZE, device=self.device, detector=self._detector)
        box = cropper.detect_box(img)
        h, w = img.shape[:2]
        if box is None:
            s = min(h, w)
            face = img[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
        else:
            x1, y1, x2, y2 = _square_box(box[0], box[1], box[2], w, h)
            face = img[y1:y2, x1:x2]
        mult = 360.0 / h
        interp = cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC
        return np.ascontiguousarray(
            cv2.resize(face, (self.SIZE, self.SIZE), interpolation=interp), dtype=np.uint8
        )

    def set_reference(self, image=None) -> float:
        """Swap the avatar's identity: re-run ``_precompute_reference`` on ``image``.

        ``image`` takes the same forms as :meth:`preprocess_reference`; ``None``
        restores ``ref_image_path`` (the process default). Returns the wall time
        in seconds.

        ``s_r`` / ``r_s`` / ``s_r_feats`` and the pose-anchor constants derived
        from them are **engine-level** state, not session state, so this must be
        sequenced on the single GPU thread strictly before the session's
        ``start_session()`` and can never overlap a ``step()``. It is not on the
        400 ms block path: measured **56 ms** end-to-end on an RTX PRO 6000 for a
        30 KB upload (SFD detect + square crop + one encode_image_into_latent),
        i.e. a sixth of the 337 ms priming block it precedes.
        """
        t0 = time.perf_counter()
        if self.G is None:
            self.load()
        self._precompute_reference(image)
        self.reference_is_default = image is None
        return time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # tensor plumbing
    # ------------------------------------------------------------------ #
    def _frames_to_tensor(self, frames: np.ndarray):
        """uint8 ``[N,512,512,3]`` RGB -> float ``[N,3,512,512]`` in [-1, 1].

        Equivalent to the offline ``A.Normalize(mean=.5, std=.5) + ToTensorV2``.
        """
        import torch

        arr = np.asarray(frames)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        t = t.permute(0, 3, 1, 2).float()
        return t.div_(127.5).sub_(1.0)

    def _to_uint8(self, img) -> np.ndarray:
        """Fixed mapping (trap (c)): never per-block min/max, which would flicker."""
        import torch

        x = img.float().clamp(-1.0, 1.0)
        x = (x + 1.0).mul_(127.5)
        x = x.permute(0, 2, 3, 1).contiguous().round_().clamp_(0, 255).to(dtype=torch.uint8)
        return x.cpu().numpy()

    def _randn(self, *shape):
        import torch

        if self._gen is not None:
            return torch.randn(*shape, device=self.device, generator=self._gen)
        return torch.randn(*shape, device=self.device)

    # ------------------------------------------------------------------ #
    # condition encoders (rolling)
    # ------------------------------------------------------------------ #
    def encode_audio_window(self, wave: np.ndarray, seq_len: int):
        """wav2vec2-encode a raw window -> ``[1, seq_len, 512]`` projected feats.

        ``wave`` must already be normalised. ``AudioEncoder.inference`` only
        replicate-pads when ``len(wave) % (seq_len*640) != 0``; we always feed
        exactly ``seq_len*640`` samples so the padding path stays dormant (trap
        6 in Agent A's report).
        """
        import torch

        a = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None].to(self.device)
        return self.G.audio_encoder.inference(a, seq_len=seq_len)

    def encode_user_motion(self, frames):
        """uint8 ``[N,512,512,3]`` (or a float tensor ``[N,3,512,512]``) -> ``[1,N,512]``."""
        import torch

        t = frames if torch.is_tensor(frames) else self._frames_to_tensor(frames)
        r_d = self.G.motion_autoencoder.enc.enc_motion(t)
        r_d = self.G.motion_autoencoder.dec.direction(r_d)
        return r_d.unsqueeze(0)

    # ------------------------------------------------------------------ #
    # the two generation primitives (verbatim sample() paths)
    # ------------------------------------------------------------------ #
    def prime_block_from_conditions(self, avatar_wa, user_wa, user_rd):
        """``sample()``'s first block: 50 frames, no KV read, ``start_pos=0``."""
        G, n = self.G, self.PRIME_FRAMES
        G.initialize_kv_cache(batch_size=1, dtype=self._kv_dtype, device=self.device)
        x_t = self._randn(1, n, self.opt.dim_w)
        pc, pwr, padaln = G.prepare_cfg_condition(
            avatar_wa, user_wa, user_rd, self.r_s, seq_len=n, context_len=0
        )
        last = len(G.denoising_step_list) - 1
        for index, ts in enumerate(G.denoising_step_list):
            x_t = G.solve_cfg(
                B=1,
                index=index,
                current_timestep=ts,
                x_t=x_t,
                precomputed_c=pc,
                precomputed_wr=pwr,
                precomputed_adaLN=padaln,
                start_pos=0,
                context_len=0,
                use_kv_cache=False,
                a_cfg_scale=self.a_cfg_scale,
                u_cfg_scale=self.u_cfg_scale,
            )
            if index == last:
                G.update_kv_cache(
                    final_latents=x_t,
                    precomputed_c=pc,
                    precomputed_wr=pwr,
                    precomputed_adaLN=padaln,
                    start_pos=0,
                )
        return x_t

    def next_block_from_conditions(self, avatar_wa, user_wa, user_rd, start_pos: int):
        """One iteration of ``sample()``'s subsequent-block loop.

        ``avatar_wa`` / ``user_wa`` / ``user_rd`` are ``[1, 12, 512]`` covering
        ``[t-2, t+10)``; returns the full ``[1, 12, 512]`` block (2 context + 10
        new latents).
        """
        import torch

        G, nb = self.G, self.BLOCK_FRAMES
        x_t = torch.cat([self._x_tail, self._randn(1, nb, self.opt.dim_w)], dim=1)
        pc, pwr, padaln = G.prepare_cfg_condition(
            avatar_wa, user_wa, user_rd, self.r_s, seq_len=nb, context_len=2
        )
        for index, ts in enumerate(G.denoising_step_list):
            x_t = G.solve_cfg(
                B=1,
                index=index,
                current_timestep=ts,
                x_t=x_t,
                precomputed_c=pc,
                precomputed_wr=pwr,
                precomputed_adaLN=padaln,
                start_pos=start_pos,
                context_len=2,
                use_kv_cache=True,
                a_cfg_scale=self.a_cfg_scale,
                u_cfg_scale=self.u_cfg_scale,
            )
        # Upstream runs update_kv_cache inside the loop under ``index == last``,
        # i.e. immediately after the final solve_cfg -- hoisting it out is
        # behaviour-identical and leaves room for the pose anchor to correct
        # ``x_t`` *before* it becomes the cached context (see _anchor_pose).
        x_t = self._anchor_pose(x_t)
        G.update_kv_cache(
            final_latents=x_t,
            precomputed_c=pc,
            precomputed_wr=pwr,
            precomputed_adaLN=padaln,
            start_pos=start_pos,
        )
        return x_t

    # ------------------------------------------------------------------ #
    # pose anchor: the drift controller
    # ------------------------------------------------------------------ #
    def _anchor_pose(self, x_t):
        """Dead-zone proportional pull of the *slow* motion component toward ``r_s``.

        Root cause of the long-session degradation (reports/drift.md): splitting
        the generated latents into a slow part (10 s temporal mean = head pose)
        and a fast part (expression / lip-sync), the fast part is stable at
        ``|z| ~ 2-8`` for 320 s while the slow part random-walks away from the
        reference motion latent ``r_s``. Measured over 800-block rollouts, a
        healthy rollout keeps ``|mu - r_s| ~ 4-5`` (cos 0.99) and 100 % sharpness;
        at ``|mu - r_s| ~ 45`` (cos 0.77) sharpness is 6 % of initial. The
        decoder consumes ``s_r + r_d``, so a wandering DC term walks that style
        vector out of its trained region -- blur, darkening, droop. The
        speaking baseline recovers to 99 % sharpness on its own whenever the
        slow state happens to wander back inside radius 5, which is what makes
        a restoring force (rather than a reset) the right correction.

        Applied to ``x_t`` before ``update_kv_cache`` / ``_x_tail``, so it acts
        on the feedback path -- it turns the marginally unstable rollout into a
        contraction instead of masking its output. Three properties make it
        imperceptible where a re-prime is not:

        * only the DC term moves, so lip-sync and expression pass through
          untouched (they live entirely in the fast part);
        * inside ``pose_deadband`` it is exactly a no-op, so ordinary pose
          variation is never fought;
        * the nudge is ramped across the block's 10 frames and is <= 0.2 per
          frame against 2-8 of natural motion, i.e. below the visible noise
          floor -- there is no event to notice, and no silent block is needed,
          so it works during continuous speech.

        Cost is one ``[1,10,512]`` reduction plus an add: <0.1 ms.
        """
        import torch

        if self.pose_gain <= 0.0:
            return x_t

        nb = self.BLOCK_FRAMES
        z = x_t[:, -nb:].float()
        m = z.mean(dim=1)                                   # [1, 512] block pose
        a = self.pose_ema
        self._pose_mu = m if self._pose_mu is None else self._pose_mu.lerp(m, a)

        d = self._pose_mu - self._r_s_row                   # [1, 512]
        dn = float(d.norm())
        self.pose_dist = dn
        if dn <= self._pose_deadband:
            return x_t

        # proportional in the *excess* only (the pull vanishes at the deadband
        # edge, so the controller has no steady-state hunting), then slew-capped.
        mag = min(self.pose_gain * (dn - self._pose_deadband), self._pose_max_step)
        corr = d * (-mag / dn)
        self.n_pose_corr += 1
        ramp = self._pose_ramp                              # [1, nb, 1], 1/nb .. 1
        out = x_t.clone()
        out[:, -nb:] = (z + corr.unsqueeze(1) * ramp).to(x_t.dtype)
        return out

    def decode_latents(self, latents) -> np.ndarray:
        """``[1, N, 512]`` motion latents -> uint8 ``[N, 512, 512, 3]`` RGB.

        ``N`` must be a multiple of ``BLOCK_FRAMES`` (decode_block's ``s_r_feats``
        are pre-expanded for exactly 10 frames).
        """
        outs = []
        n = latents.shape[1]
        for i in range(0, n, self.BLOCK_FRAMES):
            blk = latents[:, i : i + self.BLOCK_FRAMES]
            img = self.G.decode_block(
                r_d_block=blk,
                s_r=self._s_r_dec,
                s_r_feats_expanded=self._s_r_feats_exp,
                block_size=self.BLOCK_FRAMES,
                B=1,
            )
            outs.append(self._to_uint8(img))
        return np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @property
    def _kv_dtype(self):
        # trap (b): upstream allocates the cache with the *raw audio* dtype
        # (fp32) while autocast writes bf16 K/V; the cat then silently promotes
        # to fp32 and CrossAttention.append_kv_to_buffer replaces the buffer with
        # a bf16 tensor anyway. Allocating bf16 makes every read/write one dtype.
        return self.torch.bfloat16

    def start_session(self, first_user_frame: Optional[np.ndarray]) -> np.ndarray:
        """Reset all state and run the 50-frame priming block."""
        import torch

        if self.G is None:
            self.load()

        self.session_id += 1
        self.n_steps = 0
        self._t = self.PRIME_FRAMES
        self._rope_base = 0
        self.n_rope_rebase = 0
        self._pose_mu = None
        self.n_pose_corr = 0
        self._a_norm.reset()
        self._u_norm.reset()
        self._gen = None
        if self.seed is not None:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(int(self.seed))

        # 2 s of silence as the left context of both audio rings
        pad = self.PRIME_FRAMES * (self.SR // self.FPS)
        self._a_ring = np.zeros(pad, dtype=np.float32)
        self._u_ring = np.zeros(pad, dtype=np.float32)

        if first_user_frame is None:
            frame = self.ref_face
        else:
            frame = np.asarray(first_user_frame)
            if frame.shape[0] != self.SIZE or frame.shape[1] != self.SIZE:
                import cv2

                frame = cv2.resize(frame[:, :, :3], (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA)
        self._last_user_frame = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            silence = torch.zeros(1, pad, device=self.device)
            avatar_wa = self.G.audio_encoder.inference(silence, seq_len=self.PRIME_FRAMES)
            user_wa = avatar_wa.clone()
            # all 50 user frames are identical -> encode one, repeat (3.7 ms vs 106 ms)
            one = self.encode_user_motion(self._last_user_frame[None])
            user_rd = one.repeat(1, self.PRIME_FRAMES, 1)
            self._prev_user_rd = user_rd[:, -2:].clone()

            x_t = self.prime_block_from_conditions(avatar_wa, user_wa, user_rd)
            self._x_tail = x_t[:, -2:].clone()
            frames = self.decode_latents(x_t)
        return frames

    def step(
        self,
        avatar_audio: np.ndarray,
        user_audio: np.ndarray,
        user_frames: np.ndarray,
    ) -> np.ndarray:
        """Generate the next 10-frame block. Must average < 400 ms."""
        import torch

        if self._t == 0:
            raise RuntimeError("call start_session() before step()")

        a_chunk = self._as_audio(avatar_audio)
        u_chunk = self._as_audio(user_audio)
        frames = self._as_frames(user_frames)

        # --- rolling raw-audio rings (>= 2 s left context) --- #
        self._a_norm.update(a_chunk)
        self._u_norm.update(u_chunk)
        w = self._audio_window_samples
        self._a_ring = np.concatenate([self._a_ring, a_chunk])[-w:]
        self._u_ring = np.concatenate([self._u_ring, u_chunk])[-w:]
        if self._a_ring.size < w:  # only possible if start_session was skipped
            self._a_ring = np.pad(self._a_ring, (w - self._a_ring.size, 0))
            self._u_ring = np.pad(self._u_ring, (w - self._u_ring.size, 0))

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            n_ctx = 2
            need = self.BLOCK_FRAMES + n_ctx
            a_feat = self.encode_audio_window(
                self._a_norm.normalize(self._a_ring), self.audio_window_frames
            )[:, -need:]
            u_feat = self.encode_audio_window(
                self._u_norm.normalize(self._u_ring), self.audio_window_frames
            )[:, -need:]

            new_rd = self.encode_user_motion(frames)
            user_rd = torch.cat([self._prev_user_rd, new_rd], dim=1)
            self._prev_user_rd = new_rd[:, -n_ctx:].clone()

            start_pos = self._maybe_rebase_rope(self._t - n_ctx - self._rope_base)
            x_t = self.next_block_from_conditions(a_feat, u_feat, user_rd, start_pos)
            self._last_block_latents = x_t[:, -self.BLOCK_FRAMES :]
            self._x_tail = x_t[:, -n_ctx:].clone()
            out = self.decode_latents(x_t[:, -self.BLOCK_FRAMES :])

        self._t += self.BLOCK_FRAMES
        self.n_steps += 1
        self._last_user_frame = frames[-1]
        return out

    # ------------------------------------------------------------------ #
    def _as_audio(self, x) -> np.ndarray:
        a = np.asarray(x)
        if a.dtype == np.int16:
            a = a.astype(np.float32) / 32768.0
        else:
            a = a.astype(np.float32, copy=False)
        a = a.reshape(-1)
        n = self.BLOCK_SAMPLES
        if a.size < n:
            a = np.pad(a, (0, n - a.size))
        elif a.size > n:
            a = a[-n:]
        return a

    def _as_frames(self, frames) -> np.ndarray:
        n = self.BLOCK_FRAMES
        if frames is None:
            base = getattr(self, "_last_user_frame", self.ref_face)
            return np.repeat(base[None], n, axis=0)
        f = np.asarray(frames)
        if f.ndim == 3:
            f = f[None]
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        f = f[..., :3]
        if f.shape[0] < n:
            f = np.concatenate([np.repeat(f[:1], n - f.shape[0], axis=0), f], axis=0)
        elif f.shape[0] > n:
            f = f[-n:]
        if f.shape[1] != self.SIZE or f.shape[2] != self.SIZE:
            import cv2

            f = np.stack(
                [cv2.resize(x, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA) for x in f]
            )
        return np.ascontiguousarray(f, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        """Fake prime + a few fake steps so no lazy init lands on a real session."""
        import torch

        det = None
        try:  # warm the SFD detector too (its first call is slow)
            det = self._detector or get_shared_face_detector(self.device)
            flags = _cudnn_flags()
            det.detect_from_image(np.zeros((360, 480, 3), dtype=np.uint8))
            _set_cudnn_flags(flags)
        except Exception:
            pass

        rng = np.random.default_rng(0)
        frame = self.ref_face
        self.start_session(frame)
        for _ in range(max(1, self.warmup_steps)):
            a = (rng.standard_normal(self.BLOCK_SAMPLES) * 0.02).astype(np.float32)
            u = (rng.standard_normal(self.BLOCK_SAMPLES) * 0.02).astype(np.float32)
            self.step(a, u, np.repeat(frame[None], self.BLOCK_FRAMES, axis=0))
        # leave no session state behind; the next start_session() resets anyway
        self._t = 0
        self.session_id = 0
        self.n_steps = 0
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    def gpu_memory(self) -> dict:
        import torch

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 2**30,
            "reserved_gb": torch.cuda.memory_reserved() / 2**30,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        }

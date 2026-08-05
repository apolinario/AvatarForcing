"""AoT-Inductor artifacts for the AvatarForcing engine.

Shared **verbatim** between the serving Space (``multimodalart/real-avatar-streaming``)
and the compile tool (``multimodalart/avatarforcing-aoti-compile``): the compile
side exports the wrappers defined here, the serving side binds the resulting
package back onto the same call sites with :func:`aoti_loader`. Keeping one file
is what guarantees the graph that was compiled is the graph that runs.

What is compiled, and why not ``step()``
----------------------------------------
``AvatarEngine.step()`` is stateful — ``update_kv_cache`` writes into
``G.kv_cache`` in place, ``start_pos`` grows without bound and RoPE is rebased
underneath it — so it is not ``torch.export``-able as a whole. Its *leaves* are
pure functions of their inputs, though, and they are where the time goes:

``decode``       ``motion_autoencoder.dec``            Synthesis (StyleGAN2), 10 latents -> 10x3x512x512
``user_motion``  ``motion_autoencoder.enc.enc_motion`` 10x3x512x512 -> 10x20
``audio``        ``audio_encoder.inference``           1x38400 waveform -> 1x60x512

Three properties make this safe to bolt onto a live Space:

* each artifact replaces **one existing call site** with the identical
  signature, so nothing in ``server/engine.py`` changes;
* every replacement is **shape-guarded** — the compiled graph runs only for the
  exact shape it was traced at, and anything else (the 1-frame encode and the
  50-frame audio window in ``start_session``, a future block size) falls through
  to the untouched eager method;
* the loader supplies weights by *reference* and never drains the eager
  parameters (cf. ``spaces.aoti_patch`` -> ``drain_module_parameters``), so the
  eager path stays runnable — which is what makes an A/B comparison possible at
  all.

Autocast lives **inside** each wrapper's ``forward``. ``step()`` runs under
``torch.autocast("cuda", bf16)``, and an ambient autocast at export time is not
reliably captured; putting the context in the traced function bakes the same
casts into the graph. Re-entering it under the caller's autocast is a no-op.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger("avatar.aoti")

#: 10 frames per 400 ms block; the audio window is 60 feature frames (2.4 s).
#: These mirror AvatarEngine.BLOCK_FRAMES / AUDIO_WINDOW_FRAMES and are only
#: used to *name* the traced shape — the guards compare against what was
#: actually captured, not against these.
BLOCK_FRAMES = 10
AUDIO_WINDOW_FRAMES = 60


# --------------------------------------------------------------------------- #
# export wrappers
#
# Each holds the real module as ``self.m``, so every constant in the exported
# graph is named ``m.<original.fqn>`` — which is the one thing the loader has to
# reproduce.
# --------------------------------------------------------------------------- #
class DecodeWrap(nn.Module):
    """``Synthesis.forward`` with ``alpha=None`` and the feature list splatted.

    ``feats`` is a 7-element list of reference feature maps in the eager call.
    Splatting it into ``*feats`` keeps the exported input spec flat — nested
    containers survive ``torch.export`` but there is no reason to rely on it.
    """

    def __init__(self, dec: nn.Module) -> None:
        super().__init__()
        self.m = dec

    def forward(self, wa, *feats):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m(wa, None, list(feats))


class MotionWrap(nn.Module):
    """``Encoder.enc_motion``: the appearance conv stack + the motion MLP."""

    def __init__(self, enc: nn.Module) -> None:
        super().__init__()
        self.m = enc

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m.enc_motion(x)


class AudioWrap(nn.Module):
    """``AudioEncoder.inference`` for one fixed ``seq_len``.

    The replicate-pad branch of ``inference`` is dropped on purpose: the engine
    always feeds exactly ``seq_len * samples_per_frame`` samples
    (``encode_audio_window``), so the branch is dead on this path — and the
    dispatcher's shape guard is what keeps it that way.
    """

    def __init__(self, audio_encoder: nn.Module, seq_len: int) -> None:
        super().__init__()
        self.m = audio_encoder
        self.seq_len = int(seq_len)

    def forward(self, a):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feat = self.m.get_wav2vec2_feature(a, seq_len=self.seq_len)
            return self.m.audio_projection(feat)


# --------------------------------------------------------------------------- #
# dispatchers: what actually gets installed on the live module
# --------------------------------------------------------------------------- #
class _Dispatch:
    """Compiled graph for the traced shape, untouched eager method otherwise.

    Counters are cheap and are the only way to notice, in production, that a
    shape drifted and the artifact silently stopped being used.
    """

    name = "?"

    def __init__(self, compiled, eager, sig) -> None:
        self.compiled = compiled
        self.eager = eager
        self.sig = sig          # the shape signature the graph was traced at
        self.n_aoti = 0
        self.n_eager = 0

    def _fallback(self, *args, **kwargs):
        self.n_eager += 1
        return self.eager(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<aoti {self.name} aoti={self.n_aoti} eager={self.n_eager}>"


class DecodeDispatch(_Dispatch):
    name = "decode"

    def __call__(self, wa, alpha=None, feats=None):
        if alpha is None and feats is not None and _sig_decode(wa, feats) == self.sig:
            self.n_aoti += 1
            return self.compiled(wa, *feats)
        return self._fallback(wa, alpha, feats)


class MotionDispatch(_Dispatch):
    name = "user_motion"

    def __call__(self, x):
        if tuple(x.shape) == self.sig:
            self.n_aoti += 1
            return self.compiled(x)
        return self._fallback(x)


class AudioDispatch(_Dispatch):
    name = "audio"

    def __call__(self, a, seq_len):
        if (tuple(a.shape), int(seq_len)) == self.sig:
            self.n_aoti += 1
            return self.compiled(a)
        return self._fallback(a, seq_len=seq_len)


def _sig_decode(wa, feats):
    return (tuple(wa.shape), len(feats), tuple(tuple(f.shape) for f in feats))


# --------------------------------------------------------------------------- #
# the target table
#
# ``path``  — where the module lives under the AvatarForcing model ``G``
# ``attr``  — the attribute on it that the engine calls
# ``wrap``  — how to build the exportable nn.Module around it
# ``sig``   — how to derive the shape guard from a captured call
# ``args``  — how to turn a captured call into positional export args
# --------------------------------------------------------------------------- #
TARGETS: dict[str, dict] = {
    "decode": {
        "path": "motion_autoencoder.dec",
        "attr": "forward",
        "wrap": lambda mod, call: DecodeWrap(mod),
        "dispatch": DecodeDispatch,
        "sig": lambda call: _sig_decode(*_decode_parts(call)),
        "args": lambda call: (lambda wa, feats: (wa, *feats))(*_decode_parts(call)),
    },
    "user_motion": {
        "path": "motion_autoencoder.enc",
        "attr": "enc_motion",
        "wrap": lambda mod, call: MotionWrap(mod),
        "dispatch": MotionDispatch,
        "sig": lambda call: tuple(call.args[0].shape),
        "args": lambda call: (call.args[0],),
    },
    "audio": {
        "path": "audio_encoder",
        "attr": "inference",
        "wrap": lambda mod, call: AudioWrap(mod, _audio_seq_len(call)),
        "dispatch": AudioDispatch,
        "sig": lambda call: (tuple(call.args[0].shape), _audio_seq_len(call)),
        "args": lambda call: (call.args[0],),
    },
}


def _decode_parts(call):
    """``dec(s_r_d_block, alpha=None, feats=[...])`` -> ``(wa, feats)``.

    ``decode_block`` passes ``alpha``/``feats`` by keyword and ``wa``
    positionally, but capture records whatever the call site used, so both
    spellings are accepted rather than assumed.
    """
    args, kwargs = list(call.args), dict(call.kwargs)
    wa = args.pop(0) if args else kwargs.pop("wa")
    if args:
        args.pop(0)          # alpha (None on this path)
    kwargs.pop("alpha", None)
    feats = args.pop(0) if args else kwargs["feats"]
    return wa, list(feats)


def _audio_seq_len(call):
    return int(call.kwargs["seq_len"] if "seq_len" in call.kwargs else call.args[1])


# --------------------------------------------------------------------------- #
# serving side
# --------------------------------------------------------------------------- #
def _weights_for(module: nn.Module) -> dict:
    """Constants for the artifact, under every spelling it might ask for.

    Three things the obvious ``module.state_dict()`` gets wrong here:

    * it omits **non-persistent buffers** (wav2vec2 keeps some), which show up
      as "Found constant ... not provided by user" and then an illegal memory
      access on the first call;
    * it **deduplicates shared tensors**, but the traced graph refers to a
      shared buffer under *every* alias path — hence ``remove_duplicate=False``;
    * the names are relative to the wrapper used at export time, so everything
      is prefixed ``m.`` (and the dot-flattened spelling is added too, because
      artifacts refer to constants both ways depending on torch version).
    """
    weights = {}
    named = list(module.named_parameters(remove_duplicate=False))
    named += list(module.named_buffers(remove_duplicate=False))
    for name, tensor in named:
        for spelling in (f"m.{name}", f"m_{name.replace('.', '_')}", name):
            weights[spelling] = tensor
            weights[spelling.replace(".", "_")] = tensor
    return weights


def _extra_constants(path: Path, device: str):
    """Constants the live module cannot supply, shipped with the package.

    ``ToFlow.forward`` rebuilds its sampling grid with ``np.meshgrid`` on every
    call, so export lifts it as an anonymous ``_tensor_constant*`` that appears
    in no ``state_dict``. Without these the artifact loads with 1166 of 1295
    constants unset and faults on the first call.

    The move to ``"cuda"`` is deliberate and safe in the web process: ZeroGPU
    intercepts it and packs the tensors along with everything else, exactly as
    for the model's own weights. It is a few MB.
    """
    if not path.is_file():
        return {}
    blobs = torch.load(path, map_location="cpu")
    return {name: tensor.to(device) for name, tensor in blobs.items()}


def _get(module: nn.Module, path: str) -> nn.Module:
    return module.get_submodule(path) if path else module


def aoti_loader(model: nn.Module, package_dir: str, device: str = "cuda") -> None:
    """Bind every published artifact onto ``model`` (the ``AvatarForcing`` net).

    Passed to ``spaces.aoti_load(engine.G, repo_id=..., aoti_loader=...)``.
    Everything here is CPU work — download, the inductor vec-ISA probe inside
    ``LazyAOTIModel.__init__``, and binding references to the module's parameter
    tensors — so it belongs at **import in the web process**, not inside the
    lease. The binding holds references; ZeroGPU's unpack rebinds their ``.data``
    to real VRAM in the worker, so the artifact sees the restored weights with
    nothing re-transferred. (Doing the equivalent in-lease for the TTS cost
    23.7 s of a 90 s session.)

    Missing artifacts are skipped, not fatal: a package with only ``decode`` in
    it leaves the other two call sites eager.
    """
    from spaces.zero.torch.aoti import PACKAGE_FILENAME, LazyAOTIModel

    package_dir = Path(package_dir)
    installed = []
    for name, spec in TARGETS.items():
        sub = package_dir / "submodules" / name
        archive = sub / PACKAGE_FILENAME
        if not archive.is_file():
            continue
        meta = _read_sig(sub / "sig.txt")
        if meta is None:
            log.warning("aoti %s: no sig.txt, skipping (cannot guard the shape)", name)
            continue
        target = _get(model, spec["path"])
        eager = getattr(target, spec["attr"])
        weights = _weights_for(target)
        weights.update(_extra_constants(sub / "constants.pt", device))
        compiled = LazyAOTIModel(str(archive)).with_weights(weights)
        setattr(target, spec["attr"], spec["dispatch"](compiled, eager, meta))
        installed.append(name)
    log.info("aoti bound: %s", ", ".join(installed) or "nothing")
    return installed


def _read_sig(path: Path):
    """The traced shape signature, written next to the package at compile time.

    Reconstructed with ``ast.literal_eval`` rather than pickle — it is a tuple
    of ints and tuples and nothing else, so there is no reason to allow more.
    """
    import ast

    try:
        return ast.literal_eval(path.read_text().strip())
    except Exception:
        return None


def unload(model: nn.Module) -> None:
    """Put every dispatched call site back to eager. Used by the A/B bench."""
    for spec in TARGETS.values():
        target = _get(model, spec["path"])
        current = getattr(target, spec["attr"], None)
        if isinstance(current, _Dispatch):
            setattr(target, spec["attr"], current.eager)


def status(model: nn.Module) -> str:
    out = []
    for name, spec in TARGETS.items():
        current = getattr(_get(model, spec["path"]), spec["attr"], None)
        out.append(repr(current) if isinstance(current, _Dispatch) else f"<eager {name}>")
    return " ".join(out)

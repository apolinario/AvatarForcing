# Avatar Forcing — streaming server

> Upstream's paper README lives in [UPSTREAM.md](UPSTREAM.md).

A fork addition on top of [TaekyungKi/AvatarForcing](https://github.com/TaekyungKi/AvatarForcing):
a real-time conversational server around the released model. Upstream ships
`inference.py`, which is offline and whole-utterance — it encodes the entire
avatar/user waveform, then rolls the whole sequence out in one pass. This adds
an **incremental** engine that produces one 10-frame / 400 ms block per call and
can run indefinitely, plus the WebSocket server and browser client that turn it
into a two-way conversation.

Nothing about the model or the checkpoints changes. The rollout is upstream's,
one block at a time.

The conversation stack is fully local apart from the LLM: OmniVoice voice
cloning for the avatar's speech (phrase-at-a-time), Whisper large-v3-turbo for
listening, smart-turn-v3 for semantic end-of-turn (the avatar answers when you
finish a thought, not after a fixed silence), speculative transcription
(Whisper runs while the turn detector is still deciding), barge-in, and
built-in avatars with cloned voices and personas.

## Quickstart (local GPU)

```bash
git clone -b streaming https://github.com/apolinario/AvatarForcing
cd AvatarForcing
pip install -r requirements-streaming.txt
export HF_TOKEN=hf_...        # any OpenAI-compatible LLM endpoint works; the
                              # default is the HF router (see server/conversation.py)
python app.py                 # downloads checkpoints on first run, then serves :7860
```

Open http://localhost:7860, allow camera and microphone, press Start.

To run on Hugging Face (a paid dedicated-GPU Space): push this branch to a
Space with `app_file: app.py`. For free ZeroGPU hosting use the
[`streaming-zerogpu`](../../tree/streaming-zerogpu) branch, which is this
server behind a per-conversation GPU lease.

```
browser ──webcam JPEG + mic PCM──▶  server/web.py  ──▶ server/engine.py ──▶ AvatarForcing
        ◀──avatar JPEG + PCM─────                  ──▶ server/conversation.py ──▶ LLM + TTS
```

## What's here

| File | What it does |
|---|---|
| `server/engine.py` | The incremental engine. Re-formulates `AvatarForcing.sample()` as `start_session()` + repeated `step()` over rolling audio/frame buffers. Also `FaceCropper` (SFD, EMA-smoothed box). |
| `server/web.py` | `gradio.Server` app: the binary WebSocket A/V protocol, one session at a time, paced 400 ms block loop. |
| `server/conversation.py` | The conversation brain: energy VAD, barge-in, an OpenAI-compatible LLM turn, streaming TTS. |
| `static/index.html` | Self-contained browser client — capture, jitter buffer, transcript, reference-photo upload. |
| `app.py` | Entry point. Fetches checkpoints, builds the engine, serves on `:7860`. |
| `server/mock_*.py` | Engine/brain stand-ins, so the web layer can be exercised without a GPU. |

## Changes to upstream files

Three files are modified, all to bring the code up on a current environment —
no behavioural change to the model:

- **`models/wav2vec2.py`** — upstream sets `config.output_attentions = True` in
  `forward()`. transformers >= 5 raises for that unless the attention
  implementation is `eager`, and the maps are never consumed downstream (the
  audio encoder only reads `hidden_states`), so the assignment is dropped.
  `config.use_return_dict` → `config.return_dict` for the same reason.
- **`inference.py`** — `librosa` → `soundfile` + `soxr` (librosa's audio module
  needs numba, and no numba release supports the numpy this environment
  resolves to), and a small `av`-based replacement for
  `torchvision.io.write_video`, removed in torchvision >= 0.26.
- **`models/avatarforcing/AvatarForcing.py`** — optional timing instrumentation
  behind `AVATAR_TIMING=1`, off by default.

## How the incremental engine maps onto `sample()`

`sample()` does, for a T-frame utterance:

1. encode the *entire* avatar/user waveforms and all user frames;
2. block 0 (frames `[0, 50)`): fresh noise, `nfe-1` `solve_cfg` steps with
   `use_kv_cache=False`, then one `update_kv_cache`;
3. blocks `t = 50, 60, ...`: `x_t = cat(last 2 clean latents, randn(10))`,
   conditions sliced `[t-2, t+10)`, `use_kv_cache=True`, `start_pos = t-2`.

`start_session()` is step 2 with 2 s of silence and the first user frame
repeated. `step()` is exactly one iteration of step 3 — the only change is that
conditions come from rolling buffers instead of pre-computed full-utterance
tensors. Every tensor-level call (`prepare_cfg_condition` / `solve_cfg` /
`update_kv_cache` / `decode_block`) is upstream's, unmodified.

Two things had to be fixed to make an unbounded session work:

- **Rotary table.** `Attention` registers `freqs_cis` for `max_seq_len=1024`. In
  KV-cache mode `start_pos` grows without bound and
  `freqs_cis[start_pos:start_pos+12]` throws once `start_pos + 12 > 1024` — the
  session died after ~41 s. The table is rebuilt once at a configurable size and
  shared across layers; beyond that, cached keys are phase-rotated back
  (`_maybe_rebase_rope`), which is exact because RoPE is a per-position phase.
- **Rollout drift.** Sharpness decays over minutes of continuous rollout. A
  latent **pose anchor** holds the generated pose near the reference setpoint
  with a deadband relative to `|r_s|`. Measured sharpness as % of the first 20 s,
  in 20 s windows over 240 s:

  | mitigation | trace |
  |---|---|
  | none | 89 76 60 52 50 45 41 45 40 42 39 45 |
  | re-prime every 40 s | 89 76 69 63 88 75 73 55 82 73 72 54 |
  | pose anchor | 92 94 85 86 83 80 88 83 76 71 78 77 |

  The re-prime row is a sawtooth by construction — it restarts the rollout from
  the reference state, so the pose visibly snaps back. The anchor is the default;
  re-priming is kept behind `AVATAR_REPRIME_SECS` as a fallback.

## Running it

See the Quickstart above. Checkpoints are fetched from the Hub on first start
(`AVATAR_WEIGHTS_REPO`, default `multimodalart/AvatarForcingHelpers`), along
with `facebook/wav2vec2-base-960h`.

Useful knobs: `AVATAR_DEVICE`, `AVATAR_LEAD_BLOCKS`, `AVATAR_JPEG_QUALITY`,
`AVATAR_REPRIME_SECS`, `AVATAR_NORM_STD`, `AVATAR_TIMING`.

The default reference portrait is this repo's `data/rumi.jpg`; point
`AVATAR_REF_IMAGE` at another photo, or upload one in the UI.

## Deployment note

A variant of this runs on Hugging Face ZeroGPU, where the GPU only exists inside
a `@spaces.GPU` call and every call forks a fresh worker — which a stateful KV
cache cannot survive. See the `streaming-zerogpu` branch and its
[ZEROGPU.md](https://github.com/apolinario/AvatarForcing/blob/streaming-zerogpu/ZEROGPU.md):
it holds one GPU lease per conversation and drives the engine over fork queues
from the web process.

## License

Upstream is CC BY-NC 4.0 (see `LICENSE.md`) and this fork inherits it:
non-commercial, attribution required, changes indicated (this file and the
section above). The model checkpoints carry their own upstream terms.

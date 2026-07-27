# Running the streaming server on Hugging Face ZeroGPU

This branch is the [`streaming`](../../tree/streaming) server adapted to
[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu), where GPUs are
allocated per request rather than held for the process lifetime. Read the
[`streaming` branch README](../../blob/streaming/README.md) first — this only
covers the delta.

## Why it needs a delta at all

The streaming engine keeps a KV cache and rolls it forward one 400 ms block at a
time. ZeroGPU breaks both halves of what that assumes:

- **there is no GPU outside `@spaces.GPU`** — so `engine.load()`'s warm-up and
  `FaceCropper`'s SFD detections cannot run in the web process; and
- **every `@spaces.GPU` call forks a fresh worker** — module globals set in one
  call are simply absent in the next, so a per-block decorator would throw the KV
  cache away every block.

So the GPU is leased **once per conversation** instead of once per block.

## The delta

**`server/gpu_session.py`** (new) — a `@spaces.GPU` generator that forks a
worker, warms the engine, announces `__READY__` and then serves an RPC loop over
fork-context queues until the lease expires. Queues are created at *import* time,
in the parent, so the worker inherits them with their pipes intact and the parent
can keep feeding a worker that is already blocked on `get()`.

The worker owns the engine, the face cropper, the cropped-frame ring **and** JPEG
encoding. That last part is not incidental: the naive split (parent crops, ships
`[10,512,512,3]` both ways) pushes ~16 MB per block through a pickle pipe. Moving
the pixels to where they are consumed makes it ~25 KB in / ~200 KB out, and the
round trip costs ~11 ms.

**`server/engine.py`** — `load()` splits in two. ZeroGPU's parent-process patching
runs ops on CPU behind a fake-CUDA alias, so `Model(...).to("cuda")`,
`torch.load` and `param.copy_()` are all legal in the web process and get *packed*
for a fast VRAM restore — but a forward pass there would silently run on CPU.
Hence `load_weights()` (graph + parameters, no forward pass, at import) and
`warm()` (reference latents + cuDNN autotune, inside the lease).

**`server/web.py`** — the two GPU executors now submit RPCs to the leased worker
instead of touching CUDA. A session cannot start until the client holds a lease,
and any call can raise `LeaseError`, which ends the session cleanly.

**`static/index.html`** — acquires the lease through the Gradio JS client before
opening the WebSocket, shows a countdown, and offers *Keep going* / *New session*
when it expires.

## Two things that are easy to get wrong

**Bill the visitor, not the Space.** The lease is registered as a Gradio API
endpoint (`@app.api(name="run_session")`) rather than a raw route, because the
Gradio client performs the `zerogpu-headers` postMessage handshake with the
parent frame and attaches the visitor's `X-IP-Token`. Without it, GPU time falls
back to the Space's shared IP quota.

**Conversations outlive leases.** The worker is reclaimed every
`AVATAR_SESSION_SECONDS`, so the brain's turn list is carried across the boundary
(`export_history` / `import_history`) and the next lease seeds a fresh brain with
it. Only the LLM history travels — the TTS socket and the video rollout restart.

## Speech runs on the lease too

The original used ElevenLabs for both TTS and STT. This branch replaces both
with models on the same leased GPU (`server/speech.py`), so the only remote
service left is the LLM:

* **TTS — [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)**, zero-shot voice
  cloning. It is a masked diffusion LM: duration is predicted up front and a
  fixed-length sequence is denoised over `num_step` passes, so there is no
  token-level streaming to be had — the audio does not exist until the last
  step. The streaming unit is therefore the **phrase**, which is what the turn
  loop already emitted, so that loop is unchanged.
* **STT — Whisper large-v3-turbo**, one shot per VAD-delimited utterance. Whisper
  is not a streaming recogniser and the local VAD already marks the endpoint.
* **The uploaded voice clip** is transcribed by a *small* Whisper on the CPU in a
  **child process** (`server/cpu_asr.py`), before any lease exists — so it costs
  no GPU quota and the transcript is visible and editable before starting. The
  child is not an optimisation: running torch inference in the web process
  initialises CUDA there and poisons the fork.

With no uploaded clip, one voice is pinned per session by synthesising a seed
line and cloning from it. OmniVoice invents a speaker per generation, so
phrase-at-a-time synthesis would otherwise change voice mid-reply.

### AoTI

`OmniVoice.forward` is AoT-Inductor compiled (built by a separate Space, loaded
with `spaces.aoti_load` from a model repo). Measured on the half-MIG at
`num_step=32`: a ~3 s phrase drops **914 → 264 ms (3.5x)**, a 12 s one
891 → 522 ms.

Export specialises the sequence length to `8k-2` — the model pads its sequence to
a multiple of its 8 audio codebooks, so the modulo is baked into the graph.
Rather than compile per length, `PaddedAOTI` pads each call up to the next
conforming length (at most 7 positions, masked out of attention both ways) and
slices the real positions back.

Everything with a fixed cost is bound at **import**, in the web process: the
packed weights, the AoTI artifact, and the ASR weights. The lease should run the
model, not fetch and assemble it.

## Measured

On `size="large"` (half of an RTX PRO 6000 Blackwell, 48 GB):

| | |
|---|---|
| engine step | 202–231 ms |
| incl. fork round-trip | 213–239 ms |
| block budget | 400 ms |
| late blocks / dropped frames | 0 / 0 over 221 blocks |
| weights packed at import | 4.85 GB (engine + OmniVoice + Whisper) |
| in-lease warm-up | ~7 s |
| TTS phrase, eager → AoTI | 914 ms → 264 ms |

Half the card runs at ~55% of the block budget, so `size="xlarge"` buys headroom
that is already there at twice the quota cost.

## Configuration

`AVATAR_SESSION_SECONDS` (90) is *conversation* time — the clock starts after the
warm-up, and the lease requests `SESSION_SECONDS + WARM_ALLOWANCE + LEASE_MARGIN`.
Requesting the session length directly means a cold warm-up silently eats it.

Note that a large `duration` may exceed a free-tier visitor's per-call cap; lower
`AVATAR_SESSION_SECONDS` if you see `ZeroGPU illegal duration`.

Secrets: just `HF_TOKEN` — it is the only remote service left, for the LLM.
`OMNIVOICE_AOTI_REPO` points at the compiled artifact (empty string runs eager).
`requirements.txt` for a Space drops the `--extra-index-url`/`+cu128` tags and
does not list `gradio`, `spaces` or `huggingface_hub` — all platform-managed.

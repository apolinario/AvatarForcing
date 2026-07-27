# Running the streaming server on Hugging Face ZeroGPU

This branch is the [`streaming`](../../tree/streaming) server adapted to
[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu), where GPUs are
allocated per request rather than held for the process lifetime. Read
[STREAMING.md](./STREAMING.md) first — this only covers the delta.

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

## Measured

On `size="large"` (half of an RTX PRO 6000 Blackwell, 48 GB):

| | |
|---|---|
| engine step | 202–231 ms |
| incl. fork round-trip | 213–239 ms |
| block budget | 400 ms |
| late blocks / dropped frames | 0 / 0 over 221 blocks |
| weights packed at import | 1.21 GB in 5.6 s |
| in-lease warm-up | 4–13 s |

Half the card runs at ~55% of the block budget, so `size="xlarge"` buys headroom
that is already there at twice the quota cost.

## Configuration

`AVATAR_SESSION_SECONDS` (90) is *conversation* time — the clock starts after the
warm-up, and the lease requests `SESSION_SECONDS + WARM_ALLOWANCE + LEASE_MARGIN`.
Requesting the session length directly means a cold warm-up silently eats it.

Note that a large `duration` may exceed a free-tier visitor's per-call cap; lower
`AVATAR_SESSION_SECONDS` if you see `ZeroGPU illegal duration`.

Secrets: `HF_TOKEN`, `ELEVEN_TOKEN` (or `ELEVENLABS_API_KEY`), `VOICE_ID`.
`requirements.txt` for a Space drops the `--extra-index-url`/`+cu128` tags and
does not list `gradio`, `spaces` or `huggingface_hub` — all platform-managed.

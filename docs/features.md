# Feature breakdown

What the product does, grouped by what a user is trying to get done, and
staged by release. See [vision.md](vision.md) for why, and
[architecture.md](architecture.md) for how the pieces fit.

**Stages:**
- **MVP:** proves demand. It must be as easy to start as ollama and show off
  placement, which nothing else has. Runs on llama.cpp (D5), for Qwen3 dense and
  MoE. No speech model.
- **v1:** ollama parity, so people can switch without losing anything.
- **v2:** any Hugging Face model, more backends.

Draft: stages marked **?** are open questions, not decisions.

## 1. Install and first run

| Feature | Stage |
|---|---|
| Single binary (daemon + CLI), installs with one command | MVP |
| Runs as a systemd service; also runs in the foreground for trying it out | MVP |
| Detects GPUs, RAM, NUMA nodes and disks on startup, no config needed | MVP |
| Works on older NVIDIA cards (Volta / V100 and up) | MVP |
| Config file for defaults (where models live, ports, budgets) | MVP |
| Docker image | v1 |
| AMD / Apple / Intel GPUs (whatever llama.cpp supports) | v1 **?** |

## 2. Getting models

| Feature | Stage |
|---|---|
| Pull from Hugging Face by shorthand (`pull hf.co/Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M`); the tag picks the quantization, split files handled | MVP |
| Pull from community registries in a known format: OCI distribution spec, content-addressed blobs and a manifest with typed layers (GGUF, template, parameters). Any OCI registry can host models; we run no server and keep no catalog | MVP |
| Use a GGUF already on disk | MVP |
| List and remove models; show size and family | MVP |
| Choose which disk a model is stored on (this is the disk tier) | MVP |
| Modelfile equivalent: base model + defaults (prompt template, context, sampling) | v1 |
| Push models to an OCI registry | v1 |
| Search Hugging Face, ranked by what fits this machine | v1 **?** |
| Import from ollama's model store | v1 **?** |
| Non-GGUF models (safetensors) via another backend | v2 |

## 3. Serving

| Feature | Stage |
|---|---|
| OpenAI-compatible API: chat, completions, streaming | MVP |
| Several models loaded at once, each with its own placement | MVP |
| Load on first request, unload after idle (keep-alive) | MVP |
| One request at a time per model; others queue. An MVP-only shortcut: nothing else may depend on it | MVP |
| Several requests at once per model | v1 |
| Speech-to-text model (whisper.cpp), OpenAI-compatible transcription endpoint | v1 **?** |
| ollama-compatible API, so existing ollama clients work unchanged | MVP |
| Embeddings endpoint | v1 |
| No built-in auth: listens on localhost by default, the listen address is an ordinary setting, and access control belongs in a gateway in front (as with vLLM, TGI and ollama deployments) | MVP |
| API keys | later, if users ask |
| Text-to-speech (the mockup's Kokoro) | v2 **?** |
| Vision input (the mockup's vision encoders) | v2 **?** |

## 4. Placement (the differentiator)

| Feature | Stage |
|---|---|
| Model anatomy: each model broken into subsystems (attention, routed experts, shared experts, embeddings, output head, KV cache) with sizes | MVP |
| Machine view: GPUs, RAM per socket, disks, sized as in the mockup | MVP |
| Auto-place: whatever llama.cpp's own fit (`--fit`) decides for the memory that's free right now, shown and verified like any other plan | MVP |
| Edit placement by size or share per tier ("4 GiB of experts on GPU 1, 80% in RAM, rest on disk"), per D6 | MVP |
| Fit check before applying: weights, KV cache, scratch space; refuses plans that don't fit and says why | MVP |
| Apply a placement (reloads the model) | MVP |
| Verified placement: after loading, show where the weights actually are, flag anything that differs from the plan | MVP |
| Show effects users can't set: which GPU holds the expert cache, GPU 0's offload scratch space, how much may be on disk right now | MVP |
| Save placements per model and reuse them | MVP |
| KV cache placement (which GPU, how big) | v1 |
| Predicted speed for a plan before applying it | v1 **?** |
| Placement without reloading (moving a live model) | later, needs own engine |
| Per-expert placement by measured use | later, needs own engine |
| Automatic performance-aware placement per subsystem: the engine chooses where each subsystem goes (across all loaded models) to maximise speed, and adjusts as the workload changes | later, needs own engine |

## 5. Seeing what's happening

| Feature | Stage |
|---|---|
| Per model: tokens/s (prompt and generation), requests, memory per device | MVP |
| Per device: memory used by which model and subsystem | MVP |
| How much of the disk tier is actually in RAM right now | MVP |
| Expert cache hit rate | v1 |
| Prometheus metrics | v1 |
| OpenTelemetry export (metrics and request tracing) | v2 |

## 6. Interfaces

| Feature | Stage |
|---|---|
| CLI: `pull`, `run`, `list`, `rm`, `ps`, `place` | MVP |
| Web UI: the placement view from the mockup, plus model list and status | MVP |
| Interactive chat in the CLI (`run`) | v1 |
| Chat in the web UI, for trying a model | v1 **?** |

## Decisions

Decided (2026-11-15):
- No speech model in the MVP.
- One request at a time per model, in the MVP only.
- From ollama parity, the MVP gets the ollama-compatible API. The Modelfile stays
  in v1.
  Short names come from registries, not a catalog we maintain: Hugging Face
  shorthand, plus OCI registries anyone can host.

- No built-in auth; localhost by default. Production engines offer at most one
  shared key and are deployed behind gateways.

Nothing open in this document; architecture questions are in
[architecture.md](architecture.md).

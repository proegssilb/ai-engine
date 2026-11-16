# ai-engine

The most boringly named inference engine.

A placement-aware inference control plane: ollama-like ergonomics, but with
explicit, editable control over how each part of a model is spread across GPUs,
RAM, and SSD. For running serious model workloads on your own hardware.

![Placement UI mockup](docs/mockup.png)

## Start here

- **[docs/vision.md](docs/vision.md)** — the problem, the product, and the
  roadmap (MVP → ollama parity → any Hugging Face model).
- **[docs/decisions.md](docs/decisions.md)** — why it's a control plane over
  existing engines rather than a new engine, the boundary we'd own, and why the
  MVP ships on llama.cpp.
- **[docs/features.md](docs/features.md)** — feature breakdown by stage.
- **[docs/architecture.md](docs/architecture.md)** — components, the placement
  pipeline, and the decisions behind them.
- **[spikes/placement/](spikes/placement/README.md)** — a working proof that a
  Rust control plane can place each part of a model on a specific device or in
  managed RAM through llama.cpp's C API, verified on 2× V100 by checking where
  every weight actually is.

## Quick start

Install a release build (picks the build for your CPU, and the CUDA build if
`nvidia-smi` is present; checks the SHA256 sum):

```bash
curl -fsSL https://raw.githubusercontent.com/proegssilb/ai-engine/main/install.sh | bash
# or: ./install.sh --service    # system-wide, plus a systemd service (user ai-engine, data in /var/lib/ai-engine)
```

`--system` installs to `/usr/local/bin` instead of `~/.local/bin`, `--version`,
`--cpu-only` and `--level v2|v3|v4` override the choices, and rerunning
upgrades. For a private repo or a mirror, set `AIE_RELEASE_BASE` (and
`AIE_CURL_AUTH="Authorization: Bearer ..."`).

Releases come from `tools/release.sh`, one binary per x86-64 level (v2, v3 =
AVX2, v4 = AVX-512) and optionally CUDA (sm 70-89; CUDA 12.4 can't build newer).
Dev builds are tuned to the build machine; `AIE_CPU_TARGET=x86-64-v3` (or v2,
v4) makes a portable one.

Or build from source:

Linux, Rust 1.100+, cmake, clang (for bindgen), protoc. CUDA 12.x for NVIDIA
GPUs (12.x still builds for Volta / V100; CUDA 13 doesn't).

```bash
git submodule update --init
cargo build --release -p ai-engine                  # CPU only, tuned to this machine (AIE_CPU_TARGET=x86-64-v3 for a portable one)
cargo build --release -p ai-engine --features cuda  # NVIDIA; CUDA_ARCHITECTURES=70 by default

target/release/ai-engine serve &                    # API on 127.0.0.1:11435, data in ~/.ai-engine
target/release/ai-engine pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
target/release/ai-engine pull registry.example.com/team/model:q4   # or any OCI registry
curl -s localhost:11435/api/chat -d '{"model":"hf.co/Qwen/Qwen3-8B-GGUF","messages":[{"role":"user","content":"Hi"}]}'
target/release/ai-engine run hf.co/Qwen/Qwen3-8B-GGUF "Hi"   # one reply; without a prompt, a chat (Ctrl-D or /bye to leave)
target/release/ai-engine place hf.co/Qwen/Qwen3-8B-GGUF   # where every weight really is
```

The API is ollama-compatible under `/api` and OpenAI-compatible under `/v1`,
so existing clients work by pointing them at port 11435. Placement:
`GET/PUT /api/aie/placement`, or `ai-engine place MODEL --set intent.json`
(see [crates/plan](crates/plan/src/lib.rs) for the intent format). To change
several models at once, `POST /api/aie/placement/check` with
`{"edits": [{"model": ..., "intent": ... or null for automatic}]}` says what each
would put where and whether they fit together, without changing anything;
`POST /api/aie/placement/apply` with the same body applies them all, or none
if they don't fit.

Settings go in `~/.ai-engine/config.toml` (`listen`, `stores` (one per disk), `keep_alive`,
`n_ctx`, `n_batch`, `threads`, `load_timeout`) or `AIE_HOME` / `AIE_HOST`.

## Status

The MVP's serving path works end to end on CPU and is tested against real
models; the GPU path builds but hasn't run since the spike (the test box is
busy). Against the MVP list in [features.md](docs/features.md):

| Area | Done | Not yet |
|---|---|---|
| Install | one binary (daemon, worker, CLI); config file; CUDA build (links; GPU run pending); systemd unit (`contrib/systemd`); portable CPU builds (x86-64 v2/v3/v4, `AIE_CPU_TARGET`); `tools/release.sh` and `install.sh` (see below for what is verified) | CUDA release asset never built end to end; v4 build compiled but not run (no AVX-512 here) |
| Models | pull by Hugging Face shorthand and from OCI registries (verified, resumable, split models); local GGUF files; list, remove; several stores, one per disk, chosen at pull (`--store`) | registry credentials (v1) |
| Serving | ollama and OpenAI APIs with streaming; several models at once; keep-alive; one request at a time per model | |
| Placement | anatomy; auto-place (llama.cpp's fit); intents by share per tier; fit check by our estimate, then llama.cpp's dry run before loading; apply on reload; every load verified weight by weight; web UI with machine view and placement editor (`/ui/`) | KV cache placement (v1) |
| Monitoring | per model: requests, tokens, prompt and generation speed, disk-tier residency (`/api/aie/stats`, UI); per device: free memory and what each model holds (`/api/aie/machine`); the machine view's RAM bar shows the last minute of host memory (ours, free, other, disk cache; blurred by how much each border moved), per NUMA node on multi-socket machines (verified on a one-node machine only); placement panel shows per-device weights/KV/scratch, expert cache split and disk-tier residency | Prometheus (v1) |

Release tooling, verified so far: a non-native (`x86-64-v3`) build passes the
end-to-end test; `tools/release.sh` produces tarballs and `SHA256SUMS`;
`install.sh` is shellcheck-clean and tested against a local mirror (detection,
checksum, upgrade in place). Not verified: the `--service` path on a real
systemd host, the CUDA variant build, running the v2 and v4 builds, and
downloading from GitHub itself (no release exists yet).
CI (`.github/workflows/`: fmt, clippy, tests, UI check and build on every push and PR; a release workflow on `v*` tags builds the CPU levels and uploads them with `SHA256SUMS`) has never run: it is untested until first pushed, and the CUDA release build is a TODO there.

## Layout

```
crates/
  gguf/        GGUF header reader (cross-checked against llama.cpp's)
  anatomy/     model -> subsystems with sizes
  plan/        intent -> plan -> fit check -> llama.cpp settings
  store/       model store: Hugging Face pulls into content-addressed blobs
  llama/       llama.cpp backend: static build, placement, verification, generation
  proto/       daemon <-> worker gRPC protocol
  worker/      one process per loaded model
  daemon/      lifecycle, ollama/OpenAI/placement APIs
  cli/         the ai-engine binary
docs/          vision, decisions, features, architecture
fixtures/      header-only GGUFs of real models, for tests
spikes/        throwaway experiments that answer one question each
tools/         dep-age.py (how old each locked dependency is), release.sh (release tarballs)
install.sh     one-command installer
vendor/        pinned dependencies as submodules (llama.cpp)
```

Tests that need a real model use Qwen3-0.6B (`cargo run -p aie-store --example
pull -- ~/aie-store hf.co/Qwen/Qwen3-0.6B-GGUF`) and skip without it.

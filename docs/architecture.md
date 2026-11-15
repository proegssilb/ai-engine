# Architecture breakdown

How the MVP is put together. It follows the decisions in
[decisions.md](decisions.md): a Rust control plane (D3) over llama.cpp (D5),
with users placing by size and tier (D6). Features it serves are in
[features.md](features.md). Decisions taken on 2026-11-15 are listed at the end.

## Shape

```
            CLI ──┐                   ┌── Web UI (placement view)
                  ▼                   ▼
 ┌─────────────────────── daemon (supervisor) ───────────────────────┐
 │  API server ── request router ──────────────┐                     │
 │       │                                     │                     │
 │  model store   hardware inventory   lifecycle manager             │
 │       │               │                     │                     │
 │  anatomy parser ─► memory estimator ─► placement planner          │
 │                                             │ plan                │
 └─────────────────────────────────────────────┼─────────────────────┘
                                               │ IPC (spawn, plan, requests, reports)
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
                  worker: model A                             worker: model B
                  llama.cpp backend                           llama.cpp backend
                  placement verifier                          placement verifier
```

One daemon, one worker process per loaded model. A crash in llama.cpp or in our
buffer-type code takes down one model, not the control plane (D3).

## Components

### Daemon (supervisor)

| Component | Job | Notes |
|---|---|---|
| **API server** | OpenAI-compatible HTTP API, the UI's API, serves the web UI | Streaming responses |
| **Request router** | Sends requests to the right worker, loads the model first if needed, queues | MVP: one request at a time per model. Temporary; the worker protocol must allow several in flight |
| **Lifecycle manager** | Starts, stops and restarts workers; keep-alive; frees memory for a new model | Placement changes mean a reload |
| **Model store** | Downloads, lists, removes models; one store directory per disk. Sources: Hugging Face shorthand, OCI registries, local files | Content-addressed blobs, as in OCI. Where a file sits *is* its disk tier |
| **Hardware inventory** | GPUs (memory, PCIe link), RAM per NUMA node, disks; refreshed live for the UI | NVML or ggml device API; sysfs for NUMA |
| **Anatomy parser** | Reads GGUF metadata and tensor names; groups tensors into subsystems with sizes; per architecture family | Tensor names already map to subsystems (D2) |
| **Memory estimator** | Weights per subsystem + KV cache + scratch (compute buffers, op-offload scratch on GPU 0) + expert cache | Cross-check with llama.cpp's own dry run if available |
| **Placement planner** | Turns intent (sizes per tier) into a concrete plan. Auto-place in the MVP is llama.cpp's own fit (`common_fit_params`), whose output (tensor split + overrides) is already a plan in our format | The fit is in llama.cpp's C++ `common` library, not the C API: call the `llama-fit-params` tool or a few-line C wrapper |
| **State store** | Settings and saved placements as files users can edit; model index (and later metrics) in SQLite | Behind one storage interface, so SQLite can be swapped for e.g. PostgreSQL |

### Worker (one per loaded model)

| Component | Job |
|---|---|
| **Backend** | Implements the `Backend` trait over llama.cpp through FFI (whisper.cpp later) |
| **Rust RAM buffer** | The buffer type from the spike: pinned, mlocked, NUMA-bound host memory |
| **Placement verifier** | After loading, records where every weight really is (from the spike's `verify.rs`) and reports it with the plan's differences |
| **Metrics reporter** | Throughput, memory per device, disk-tier residency, cache hit rate |

## The placement pipeline

```
 user intent                 concrete plan                     backend settings
 ───────────                 ─────────────                     ────────────────
 "experts: 10% GPU 1,        layers 0-4 experts → CUDA1        tensor_buft_overrides
  80% RAM, 10% disk"   ──►   layers 5-42 experts → pinned ──►  (regex → buffer type),
 "attention: GPU 0"          layers 43-47 experts → mmap        n_gpu_layers, tensor_split,
                             attention → CUDA0                  moe_cache_size, no_host,
                             expert cache: 0                    op_offload, use_mmap
                             budget check: fits ✓
```

1. **Intent** is what the UI edits: per subsystem, a size or share per tier
   (D6).
2. **The planner** turns shares into whole layers (or an expert-cache size), adds
   KV cache and scratch, and checks the fit on every device. It returns either
   a plan or the reason it doesn't fit.
3. **The backend compiles** the plan to its own settings. It makes one
   deliberate choice, "allow CPU repacking or not", which sets `extra_bufts` and
   `no_host` together: repacking only happens with both on, and it copies the
   affected weights out of the placement the rule asked for (D4). Otherwise
   `no_host` stays at its default; with mmap on it changes nothing.
4. **The worker loads, verifies and reports.** The UI shows the verified
   placement, not the plan. A mismatch is a bug, and the same check runs as a
   regression test when llama.cpp is upgraded.

## The `Backend` trait (sketch)

```rust
trait Backend {
    /// What this backend can place, and how finely (whole layers, cache size, ...).
    fn capabilities(&self) -> Capabilities;
    /// Plan → backend settings; errors if the plan asks for something unsupported.
    fn compile(&self, plan: &Plan) -> Result<BackendConfig>;
    fn load(&mut self, model: &ModelFile, cfg: &BackendConfig) -> Result<Loaded>;
    fn verify(&self, loaded: &Loaded) -> PlacementReport;
    fn generate(&self, loaded: &Loaded, req: Request) -> Stream<Token>;
    fn metrics(&self, loaded: &Loaded) -> Metrics;
}
```

The UI greys out what `capabilities()` says the backend can't do (D3).

## Request path

`client → API server → router → worker (IPC) → backend → tokens streamed back the same way`

## Starting a model

`router sees a request for an unloaded model → lifecycle manager picks the
saved or auto plan → memory estimator checks it fits (unloading idle models if
needed) → worker spawned with the plan → load → verify → report → requests flow`

## What comes from the spike

- **Rust RAM buffer** (`spikes/placement/src/pinned.rs`), bindings via bindgen,
  the llama.cpp build via the `cmake` crate.
- **Placement verifier** (`spikes/placement/src/verify.rs`), with the fixes Fable's
  review found: count only selected experts in copy volume, don't double-count
  expert-cache tensors, attribute copied tensors to their original weight.
- **The list of llama.cpp settings that must be set deliberately** (D4).

## Decisions

Decided (2026-11-15):
- **IPC between daemon and workers: gRPC (tonic) over a Unix socket.** Runtime
  overhead is negligible next to token times (microseconds against 7–70 ms per
  token), and nothing large crosses the connection. It brings streaming,
  cancellation and many requests in flight, which lifting the MVP's
  one-request limit needs.

- **Web UI: a TypeScript single-page app**, built to static files and embedded in
  the binary. The placement editor's drag-and-drop needs rich client-side
  interaction.

- **State: files for what users edit, SQLite for what the daemon indexes.**
  Settings and saved placements are readable files users can edit and version;
  SQLite holds the model index and, later, metrics history. SQLite must be
  swappable for a server database (e.g. PostgreSQL): all database access goes
  through one storage interface, queries stay in portable SQL, and the tests
  run against both.

- **Memory estimation: both, cross-checked.** Our own arithmetic answers the UI
  instantly while users drag; llama.cpp's dry run (`no_alloc`) is the
  authoritative check before applying; the verified numbers after loading
  correct our arithmetic. A disagreement is a bug. (llama.cpp's per-device
  breakdown is in an internal header, `llama-ext.h`, or in the logs, so read it
  with the same suspicion as the placement log.)

- **One worker process per model** (D3). Loading spawns, unloading exits, so
  memory comes back cleanly and a crash takes down one model. Each process pays
  for a CUDA context on every GPU it uses (typically a few hundred MiB; not yet
  measured on the V100s), which the memory estimator includes.

Consequence to live with, not a choice: **llama.cpp stays pinned.** The Rust
buffer depends on an internal header (`ggml-backend-impl.h`), so every upgrade
means regenerating bindings and running the placement regression test.

## As built (2026-11-15)

Where the first implementation refined the design above:

- **The daemon never initialises CUDA.** Anything that needs the devices runs
  in a worker: llama.cpp's fit (auto-place), and planning a saved intent
  (which needs the GPU count and free memory). A worker therefore takes one of
  three placements: `Auto` (llama.cpp's fit, resolved into a plan), `Intent`
  (planned, fit-checked and compiled in the worker; refused with the reason if
  it doesn't fit), or `Planned` (already compiled). Every load is then verified
  weight by weight against its plan, and the worker's `Status` carries both.
- **The fit shim is the only C++** (`crates/llama/shim/fit.cpp`), linked into
  the worker, as decided in D3's exception.
- **Saved placements are files** (`AIE_HOME/placements/<model>.json`, the
  intent as JSON). The model index is the store's manifests for now; SQLite
  arrives with the first thing that needs queries (metrics history).
- **Worker sockets live in `$XDG_RUNTIME_DIR/ai-engine` or `/tmp/aie-<uid>`**,
  not under `AIE_HOME`: Unix socket paths max out at about 108 bytes.
- **llama.cpp links statically, in dependency order**, through
  `rustc-link-lib` (link arguments don't reach dependent crates). The CUDA
  runtime and cuBLAS stay dynamic.
- **Machine view without CUDA in the daemon:** a short-lived `ai-engine probe`
  subprocess reports devices as JSON (cached a few seconds).
- **Memory: estimate, then dry run.** The worker checks an intent with our
  estimate first (instant, also what the UI shows), then runs llama.cpp's dry
  run (`common_fit_measure`, `no_alloc`) with the exact load parameters and
  refuses the plan if a device would overflow. The measured per-device
  breakdown is in the worker's status. llama.cpp's fit and dry run swap the
  global logger, so the shim's two calls are serialised per process.
- **The shim has two functions:** the fit and the dry run, both from
  llama.cpp's `common` library.

How each placement promise in the UI is held and checked (weights, RAM tier, disk tier, host socket): [reference/placement-guarantees.md](reference/placement-guarantees.md).

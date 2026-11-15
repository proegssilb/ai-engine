# Decision log

Short records of the choices that shape the project, each with the reasoning
and current status. Newest decisions at the top. See [vision.md](vision.md) for
the product direction these serve.

---

## D6 — Users place by size and tier; the engine handles the rest

**Status:** adopted (2026-11-15).

Users say how much of each subsystem goes where, by size or share: "this 16 GiB
GPU holds 4 GiB of experts, 80% in RAM, 10% on disk". They don't pick
individual experts, and they don't pick memory flavours.

**Why:**
- **Users can't know which experts matter.** Choosing hot experts belongs to
  the engine (llama.cpp's MoE cache does it today; a frequency-based planner
  could later).
- **Size-based control is available on llama.cpp.** By whole layers (one
  layer ≈ 2% of experts on a 48-layer model) or by MoE-cache size. Both were
  verified on the GPU box.
- **Pinned vs. pageable host memory isn't a user choice.** Measured on 2× V100:
  pinned memory is about 3% faster on prompt processing and identical on decode.
  The engine picks pinned by default.
- **The disk tier is a performance floor, not a guarantee.** Memory-mapped
  pages may be dropped from RAM, never forced out. That's explainable in the UI.

**What the UI shows but doesn't let users set:** which GPU holds the MoE cache
(it follows the layer split), the op-offload scratch space on GPU 0, and run-time
copies.

---

## D5 — Ship the MVP on llama.cpp; build our own engine after demand is proven

**Status:** adopted (2026-11-15). Supersedes the "pending the GPU benchmark"
status of D4.

The MVP runs on llama.cpp. A custom engine waits until the product has proven
there's demand for it.

**Evidence (GPU placement spike, 2026-11-15, Qwen3-30B-A3B on 2× V100):**
- **Placement rules hold at rest** once a few options are set deliberately:
  Rust-pinned RAM, an explicit GPU, and the mmap'd file all hold the bytes the
  rule said, confirmed by checking each weight's buffer and address.
- **The leaks are few and known.** See D4 for the list. Each one is either
  fixed by a setting or only affects speed.
- **Our placement costs nothing extra.** Weights in Rust-pinned RAM ran the same
  as llama.cpp's own CPU placement.
- **No other engine is clearly better on this hardware.** vLLM and SGLang offer
  coarser offload (a GB budget or layer groups), and recent vLLM reportedly
  dropped V100. KTransformers has the most expressive MoE placement (YAML rules,
  hot experts by frequency), but its current release needs Ampere or newer.
  mistral.rs maps devices per layer range. ik_llama.cpp uses the same override
  mechanism, so it has the same leaks. HF Accelerate honours a per-module map
  including disk, but it isn't a serving engine. Rule-based placement over a
  scheduler with its own policy leaks everywhere.

**What would revisit this:** demand proven, or a placement users need that
llama.cpp can't express (per-expert control, a chosen disk tier), or pipeline
parallelism turning out to matter for multi-user serving.

---

## D4 — The boundary we own: loader + memory manager, not the kernels

**Status:** the boundary for a future engine; deferred until after the MVP (D5).

If we do grow our own engine code, the line is: **own the model loader and the
memory manager; borrow the compute kernels.** We read the model files, decide
where every tensor (eventually every expert) lives, manage the GPU/RAM/SSD tiers
and the KV cache, and move data between tiers. The matrix math is handed to an
existing kernel library.

**Why:** the placement spike showed the two halves of llama.cpp behave very
differently for us. The compute side just works — the same model math produced
bit-identical output wherever the weights lived. What resists control is the
loader and memory management: llama.cpp relocates weights on its own (it moved
attention weights into an AMX buffer unasked), the SSD tier's contents are the
kernel's page-cache LRU rather than our choice, experts can only be pinned a
whole tensor at a time, and placement is fixed at load. That list is a
memory-manager's job description. The compute kernels are not where our value or
our friction is.

The GPU run (2026-11-15) added more of the same, all in the loader, scheduler
or cache, none in the kernels:
- **The debug log misreports CPU rules.** A `CPU` rule is logged as going to
  `CUDA_Host`, then silently moved to the mmap'd buffer. Placement has to be
  checked by address, not read from the log.
- **Without mmap, `CPU` becomes CUDA-pinned host memory** unless `no_host` is
  set. Harmless: 3% faster prompt processing, and it doesn't happen with mmap
  on, which the disk tier needs anyway.
- **`extra_bufts` with `no_host` silently copies experts into a repack buffer.**
  Without `no_host`, `CUDA_Host` wins and `extra_bufts` does nothing. So the two
  are one deliberate choice: allow CPU repacking or not.
- **Op offload:** for batches of 32+ tokens, the experts each batch selects are
  copied from RAM to GPU 0 for every batch, through about 140 MiB of
  scratch space reserved at startup. It's on or off for the whole context. On
  average it helps (203 vs. 161 tok/s on a 512-token prompt), so leave it on and
  budget for it.
- **The MoE cache stands in for the placed experts** on batches of up to 32
  tokens; misses are still read from where the rule put them.
- **Any placement rule disables pipeline parallelism.** No measurable cost on
  single-user runs; unmeasured for larger batches or several users.

**Caveat still open:** owning the loader means taking on per-model architecture
code (attention variants, expert routing, rope scaling), which is a large part
of llama.cpp's value. For the MVP's two families that's acceptable; past that
it's a real cost. The GPU benchmark decides how soon this matters.

---

## D3 — Rust control plane around a `Backend` trait; llama.cpp via FFI

**Status:** adopted; spiked.

The engine is a Rust control plane. Backends sit behind a trait; the first
backend is llama.cpp, used as a library through its C API (no C++ written).
Each backend reports what placement it can actually honor, and the UI greys out
what it can't.

**Why Rust, not contributing to llama.cpp:** extending llama.cpp itself means
C++, which we're not taking on for this project. Its contribution policy also
requires disclosing AI-assisted code, a full manual review ("one hour per
200-400 LOC"), being able to explain every line, and bars AI-written PR text —
so AI-heavy contribution isn't a fit either. A private C++ fork has the same
maintenance problem long-term. Rust with llama.cpp behind a C ABI gives us
control without any of that.

**Isolation:** FFI in-process means a crash in llama.cpp or in our buffer-type
callbacks takes down the control plane. The intended shape is a small worker
process per model that links the backend, plus a separate supervisor — library
control with process isolation. The spike runs in-process for simplicity.

---

## D2 — Placement is a compiler target

**Status:** adopted; spiked.

The UI's "subsystem → device" layout is the source language; the backend's
placement controls are the target. The planner compiles one to the other.

**Why it works:** GGUF tensor names already line up with the subsystem boxes in
the mockup — `token_embd`, `blk.N.attn_*`, `blk.N.ffn_gate_inp` (router),
`blk.N.ffn_*_exps` (routed experts), `blk.N.ffn_*_shexp` (shared experts),
`output`. A box dragged onto a device becomes an ordered list of (name-pattern →
destination) rules, which is exactly the shape llama.cpp's `-ot` /
`tensor_buft_overrides` already takes. The spike drives this from Rust and
checks the resulting placement by recording each weight's real buffer and
address during evaluation. llama.cpp's own logs proved unreliable for this.

**Three tiers for weights,** established by the spike:
1. **GPU** — the backend's device buffer types.
2. **RAM we own** — a buffer type we implement (NUMA node, mlock, hugepages,
   optional CUDA host-register under our control).
3. **SSD** — the backend's default CPU buffer, memory-mapped, used as a cache
   whose contents the kernel decides, not us. It's a performance floor, not a
   guarantee: those pages may be dropped from RAM, never forced out (D6).

Controlling tier 3's contents deliberately (real NVMe expert streaming) is the
first thing a backend can't express, and the likeliest first reason to write
engine code of our own.

---

## D1 — A control plane, not a platform or a from-scratch engine

**Status:** adopted.

Build a placement-aware control plane over existing engines, not a new
inference engine and not a full deployment platform. See
[vision.md](vision.md) for the market gap this fills.

**Why not a from-scratch engine:** the compute core is enormous, well-served by
existing libraries, and not our differentiator. **Why not a platform:** the
point is ollama-like "just run it" ergonomics with real placement control, not
a kit to assemble or an appliance that hides the knobs. The differentiator is
the placement model and its UI, which is a thick wrapper, not a new engine.

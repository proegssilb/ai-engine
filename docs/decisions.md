# Decision log

Short records of the choices that shape the project, each with the reasoning
and current status. Newest decisions at the top. See [vision.md](vision.md) for
the product direction these serve.

---

## D4 — The boundary we own: loader + memory manager, not the kernels

**Status:** working hypothesis, pending the GPU benchmark.

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
reconstructs the resulting placement from llama.cpp's own logs.

**Three tiers for weights,** established by the spike:
1. **GPU** — the backend's device buffer types.
2. **RAM we own** — a buffer type we implement (NUMA node, mlock, hugepages,
   optional CUDA host-register under our control).
3. **SSD** — the backend's default CPU buffer, memory-mapped, used as a cache
   whose contents the kernel decides, not us.

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

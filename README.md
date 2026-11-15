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
  existing engines rather than a new engine, and the boundary we'd own.
- **[spikes/placement/](spikes/placement/README.md)** — a working proof that a
  Rust control plane can place each part of a model on a specific device or in
  managed RAM through llama.cpp's C API, with bit-identical output. Includes a
  benchmark ready to run on a real GPU box.

## Status

Early. The direction and the core feasibility question are settled and
documented; the engine itself isn't built yet. The next concrete step is the
GPU placement benchmark described in the spike README, which decides how much
of the engine we end up owning.

## Layout

```
docs/          vision and decision log
spikes/        throwaway experiments that answer one question each
  placement/   per-tensor placement through llama.cpp, from Rust
vendor/        pinned dependencies as submodules (llama.cpp)
```

Spikes are deliberately not production code; each one's README says what it
proved and what it skipped.

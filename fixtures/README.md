# Test fixtures

Header-only GGUF files: real metadata and tensor tables, no weights. Made with
`cargo run -p aie-gguf --example gguf-fixture -- SOURCE FIXTURE`, which drops the
tokenizer arrays and checks the tensor table is preserved exactly.

| File | Source |
|---|---|
| `qwen3-30b-a3b-q4_k_m.header.gguf` | First 32 MiB of `Qwen/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf` (Apache-2.0). Tensor table checked identical to llama.cpp's reader with `crates/gguf/xcheck/run.sh`. |
| `qwen3-8b-q4_k_m.header.gguf` | First 16 MiB of `Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` (Apache-2.0). Tensor table checked identical to llama.cpp's reader. |

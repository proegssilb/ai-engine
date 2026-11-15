#!/usr/bin/env bash
# Cross-check aie-gguf against llama.cpp's own GGUF reader, field by field.
# Needs the llama.cpp libraries the placement spike builds
# (cd spikes/placement && cargo build --release).
# Usage: crates/gguf/xcheck/run.sh FILE.gguf [FILE.gguf ...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
LIB=$(dirname "$(find "$ROOT/spikes/placement/target/release/build" -name 'libggml-base.so' | head -1)")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
gcc "$ROOT/crates/gguf/xcheck/gguf-dump.c" -I"$ROOT/vendor/llama.cpp/ggml/include" -L"$LIB" -lggml-base -Wl,-rpath,"$LIB" -o "$TMP/llama-dump"
cargo build -q --manifest-path "$ROOT/Cargo.toml" -p aie-gguf --example gguf-dump
for f in "$@"; do
  "$TMP/llama-dump" "$f" > "$TMP/llama.txt"
  "$ROOT/target/debug/examples/gguf-dump" "$f" > "$TMP/ours.txt"
  if diff -q "$TMP/ours.txt" "$TMP/llama.txt" > /dev/null; then
    echo "identical: $f ($(($(wc -l < "$TMP/ours.txt") - 1)) tensors)"
  else
    echo "DIFFERENT: $f"; diff "$TMP/ours.txt" "$TMP/llama.txt" | head -20; exit 1
  fi
done

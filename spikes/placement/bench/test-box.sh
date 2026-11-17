#!/usr/bin/env bash
# Placement benchmark for the test box (2x V100, Ryzen 5800X3D, 64 GB RAM, 1 NVMe).
#
# Compares llama.cpp's own placement (llama-bench with -ngl / -ot) against the
# same placements driven through the C API by placement-spike, then tries the
# placements only the spike can do: experts in Rust-pinned RAM, and the
# mmap/NVMe tier under a memory cap.
#
# Usage:
#   bench/test-box.sh /path/to/Qwen3-30B-A3B-Q4_K_M.gguf [scenario ...]
# Scenarios (default: all but pressure):
#   ref        llama-bench reference numbers
#   gpu        everything on the GPUs
#   experts    attention on GPU, routed experts on the CPU side, three ways
#   mixed      half the experts on GPU, half pinned in RAM
#   moecache   llama.cpp's moe_cache_size GPU expert cache over each CPU tier
#   pressure   memory-capped runs that force expert streaming from NVMe
#              (needs systemd-run --user with memory delegation, or sudo)
#
# Env knobs: PROMPT (512), GEN (128), THREADS (8), CUDA_ARCHITECTURES (70),
#            MEMCAP (10G, pressure scenario), PIN_LAYERS (regex, pressure scenario)
set -euo pipefail

MODEL=${1:?usage: $0 MODEL.gguf [scenario ...]}
shift
SCENARIOS=${*:-ref gpu experts mixed moecache}
PROMPT=${PROMPT:-512}
GEN=${GEN:-128}
THREADS=${THREADS:-8}   # 5800X3D: 8 physical cores; SMT siblings rarely help matmul
export CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-70}

HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT="$HERE/results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
SPIKE="$HERE/target/release/placement-spike"
LLAMA_BENCH="$REPO/vendor/llama.cpp/build-cuda/bin/llama-bench"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- prereqs --
command -v nvcc >/dev/null || { echo "nvcc not found: install a CUDA 12.x toolkit"; exit 1; }
CUDA_MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
if [ "${CUDA_MAJOR:-0}" -ge 13 ] && [ "$CUDA_ARCHITECTURES" = 70 ]; then
  echo "CUDA $CUDA_MAJOR can't build for Volta (sm_70). Use a CUDA 12.x toolkit for the V100s."; exit 1
fi
git -C "$REPO" submodule update --init vendor/llama.cpp

{
  echo "date: $(date -Is)"; echo "model: $MODEL ($(du -h "$MODEL" | cut -f1))"
  echo "llama.cpp: $(git -C "$REPO/vendor/llama.cpp" rev-parse --short HEAD)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
  lscpu | grep -E 'Model name|^CPU\(s\)|NUMA node\(s\)|L3'
  free -g; nvcc --version | tail -1
} | tee "$OUT/system.txt"

log "build placement-spike (CUDA)"
(cd "$HERE" && cargo build --release --features cuda) 2>&1 | tail -3

if [ ! -x "$LLAMA_BENCH" ]; then
  log "build llama-bench (CUDA) for reference numbers"
  cmake -S "$REPO/vendor/llama.cpp" -B "$REPO/vendor/llama.cpp/build-cuda" -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$REPO/vendor/llama.cpp/build-cuda" --target llama-bench -j"$(nproc)" 2>&1 | tail -2
fi

# ---------------------------------------------------------------- runners --
EXP='ffn_(up|down|gate)_exps'
PREFIX=()   # e.g. a systemd-run memory cap

spike() { # spike LABEL [args...]
  local label=$1; shift
  log "spike: $label${PREFIX[*]:+ (under: ${PREFIX[*]})}"
  ${PREFIX[@]+"${PREFIX[@]}"} "$SPIKE" "$MODEL" --label "$label" --prompt-tokens "$PROMPT" --tokens "$GEN" --threads "$THREADS" "$@" \
    2>&1 | tee "$OUT/$label.log" | grep -E '^(devices|  CUDA|  CPU|loaded|  llama.cpp|  RustPinned|prefill|cold:)' || true
  grep '^RESULT ' "$OUT/$label.log" | sed 's/^RESULT //' >> "$OUT/results.jsonl" || echo "  (no RESULT; see $OUT/$label.log)"
}

refbench() { # refbench LABEL [llama-bench args...]
  local label=$1; shift
  log "llama-bench: $label"
  "$LLAMA_BENCH" -m "$MODEL" -p "$PROMPT" -n "$GEN" -t "$THREADS" -r 3 -o jsonl "$@" 2>"$OUT/$label.err" \
    | tee "$OUT/$label.jsonl" | python3 -c '
import json,sys
for l in sys.stdin:
    r=json.loads(l); print(f"  {\"pp\" if r[\"n_prompt\"] else \"tg\"}: {r[\"avg_ts\"]:.2f} tok/s ± {r[\"stddev_ts\"]:.2f}")' || echo "  failed; see $OUT/$label.err"
}

# -------------------------------------------------------------- scenarios --
for s in $SCENARIOS; do case $s in
ref)
  refbench ref-all-gpu      -ngl 99
  refbench ref-experts-cpu  -ngl 99 -ot "exps=CPU"
  ;;
gpu)
  # Sanity check: should match ref-all-gpu, or the C API path itself costs something.
  spike gpu-all --gpu-layers -1
  ;;
experts)
  # The common "big MoE on small GPUs" recipe, three ways to hold the experts in host RAM.
  spike exp-mmap         --gpu-layers -1 --place "$EXP=CPU"
  spike exp-mmap-repack  --gpu-layers -1 --place "$EXP=CPU" --extra-bufts   # = ref-experts-cpu
  spike exp-pinned       --gpu-layers -1 --place "$EXP=pinned"
  spike exp-pinned-reg   --gpu-layers -1 --place "$EXP=pinned" --cuda-register
  ;;
mixed)
  # First half of the layers keep their experts on the GPUs, the rest are pinned in RAM.
  spike mixed-pinned     --gpu-layers -1 --place "blk\.(2[4-9]|3[0-9]|4[0-7])\.$EXP=pinned" --cuda-register
  spike mixed-mmap       --gpu-layers -1 --place "blk\.(2[4-9]|3[0-9]|4[0-7])\.$EXP=CPU" --extra-bufts
  ;;
moecache)
  spike moecache-mmap    --gpu-layers -1 --place "$EXP=CPU" --extra-bufts --moe-cache 4096
  spike moecache-pinned  --gpu-layers -1 --place "$EXP=pinned" --cuda-register --moe-cache 4096
  ;;
pressure)
  # Cap the process's memory below the expert footprint so the mmap tier has to
  # stream from NVMe. Page cache counts against the cap; so does mlocked memory.
  MEMCAP=${MEMCAP:-10G}
  PIN_LAYERS=${PIN_LAYERS:-'blk\.([0-9]|1[0-5])\.'}   # ~1/3 of the experts
  if systemd-run --user --scope -q -p MemoryMax=1G true 2>/dev/null; then
    PREFIX=(systemd-run --user --scope -q -p MemoryMax="$MEMCAP" -p MemorySwapMax=0)
  else
    PREFIX=(sudo systemd-run --scope -q -p MemoryMax="$MEMCAP" -p MemorySwapMax=0)
  fi
  spike pressure-mmap        --gpu-layers -1 --place "$EXP=CPU" --evict
  spike pressure-pin-third   --gpu-layers -1 --place "$PIN_LAYERS$EXP=pinned" --place "$EXP=CPU" --cuda-register --evict
  PREFIX=()
  ;;
*) echo "unknown scenario $s"; exit 2;;
esac; done

log "summary"
python3 "$HERE/bench/summarize.py" "$OUT" | tee "$OUT/summary.md"
echo "results in $OUT"

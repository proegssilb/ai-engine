#!/usr/bin/env bash
# Memory spike, phase C: keep the RAM tier out of swap without mlock and
# without root: a systemd *user* scope with MemorySwapMax=0. Default 8 MiB
# memlock limit, no CUDA registration. Same pressure as phase B.
set -u
BIN=/srv/ollama/spike/mem-target/release/memory-spike
M=/srv/ollama/spike/models/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=/srv/ollama/spike/memory-results; mkdir -p $OUT
SWAP=/srv/ollama/aie-spike-swapfile
COMMON="--ram-share 0.5 --threads 8 --prompt 512 --tokens 64 --repeat 3 --no-mlock"
cleanup() { sudo swapoff $SWAP 2>/dev/null; sudo rm -f $SWAP; echo "swap after cleanup: $(grep SwapTotal /proc/meminfo)"; }
trap cleanup EXIT
sudo fallocate -l 32G $SWAP && sudo chmod 600 $SWAP && sudo mkswap -q $SWAP && sudo swapon $SWAP
grep SwapTotal /proc/meminfo; echo "memlock limit: $(ulimit -l) KiB (not raised)"
run() { local label=$1 swapmax=$2
  echo "== $label (user scope, MemoryMax=12G, MemorySwapMax=$swapmax)"; sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  timeout 1800 systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=$swapmax $BIN $M $COMMON --label $label > $OUT/$label.out 2> $OUT/$label.err
  echo "exit=$?"
  mkdir -p $OUT/one && cp $OUT/$label.out $OUT/$label.err $OUT/one/ && python3 /srv/ollama/spike/mem-sum.py $OUT/one; rm -rf $OUT/one
}
run c1-user-scope-swap-allowed   infinity
run c2-user-scope-swap-forbidden 0
echo PHASE_C_DONE

#!/usr/bin/env bash
# Memory spike, phase B: memory pressure with swap. Half the experts in the
# RAM tier, half in the disk tier; process capped so the kernel must evict.
# Temporary swap file, removed at the end.
set -u
BIN=/srv/ollama/spike/mem-target/release/memory-spike
M=/srv/ollama/spike/models/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=/srv/ollama/spike/memory-results; mkdir -p $OUT
SWAP=/srv/ollama/aie-spike-swapfile
COMMON="--ram-share 0.5 --threads 8 --prompt 512 --tokens 64 --repeat 3"
AS_ME="prlimit --memlock=unlimited:unlimited setpriv --reuid=minifig --regid=minifig --init-groups"
cleanup() { sudo swapoff $SWAP 2>/dev/null; sudo rm -f $SWAP; echo "swap removed: $(swapon --show | wc -l) swap devices"; }
trap cleanup EXIT
sudo fallocate -l 32G $SWAP && sudo chmod 600 $SWAP && sudo mkswap -q $SWAP && sudo swapon $SWAP
swapon --show
run() { local label=$1 cap=$2; shift 2
  echo "== $label (cap $cap)"; sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  if [ "$cap" = none ]; then
    timeout 1800 sudo $AS_ME $BIN $M $COMMON --label $label "$@" > $OUT/$label.out 2> $OUT/$label.err
  else
    timeout 1800 sudo systemd-run --scope -q -p MemoryMax=$cap -p MemorySwapMax=infinity $AS_ME $BIN $M $COMMON --label $label "$@" > $OUT/$label.out 2> $OUT/$label.err
  fi
  echo "exit=$?"
  mkdir -p $OUT/one && cp $OUT/$label.out $OUT/$label.err $OUT/one/ && python3 /srv/ollama/spike/mem-sum.py $OUT/one; rm -rf $OUT/one
  tail -2 $OUT/$label.err | grep -iE "error|panic|kill" || true
}
run b0-reference        none
run b1-unlocked-cap12G  12G --no-mlock
run b2-locked-cap12G    12G
run b3-cudareg-unlocked-cap12G 12G --no-mlock --cuda-register
# reserved 2 MiB pages: never swapped, but outside the memory cap's accounting
echo 4400 | sudo tee /proc/sys/vm/nr_hugepages >/dev/null; grep HugePages_Total /proc/meminfo
run b4-hugetlb-unlocked-cap12G 12G --no-mlock --hugetlb 2M
echo 0 | sudo tee /proc/sys/vm/nr_hugepages >/dev/null
echo PHASE_B_DONE

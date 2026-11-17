#!/usr/bin/env bash
# RAM-tier reporting on the GPU box: (1) defaults, (2) swap on and the
# user-scope backstop turned off, so only CUDA registration holds it.
set -u
BIN=/srv/ollama/spike/mem-target/release/ai-engine
M=/srv/ollama/spike/models/Qwen3-30B-A3B-Q4_K_M.gguf
H=/srv/ollama/spike/aie-home; rm -rf $H; mkdir -p $H/store
SWAP=/srv/ollama/aie-spike-swapfile
cleanup() { pkill -f "[a]i-engine serve"; pkill -f "[a]i-engine worker"; sleep 2; sudo swapoff $SWAP 2>/dev/null; sudo rm -f $SWAP; rm -rf $H; echo "cleanup: $(grep SwapTotal /proc/meminfo), $(pgrep -c -f '[a]i-engine') ai-engine processes"; }
trap cleanup EXIT
INTENT='{"n_ctx":2048,"layers":[{"tier":{"Gpu":0},"amount":{"Fraction":0.5}},{"tier":{"Gpu":1},"amount":"Rest"}],"subsystems":{"RoutedExperts":[{"tier":"Ram","amount":{"Fraction":0.5}},{"tier":"Disk","amount":"Rest"}]},"embeddings":null,"output":null,"expert_cache_bytes":0,"allow_repack":false}'
check() { local title=$1
  AIE_HOME=$H AIE_HOST=127.0.0.1:11496 $BIN serve > $H/daemon.log 2>&1 &
  sleep 3; head -1 $H/daemon.log
  curl -s -X PUT localhost:11496/api/aie/placement -d "{\"model\":\"$M\",\"intent\":$INTENT}" >/dev/null
  curl -s localhost:11496/api/generate -d "{\"model\":\"$M\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":8}}" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("generate:", d.get("done_reason"), d.get("error",""))'
  echo "== $title"; $BIN --host 127.0.0.1:11496 place $M | sed -n '1,8p'
  pkill -f "[a]i-engine serve"; sleep 3
}
echo "memlock: $(ulimit -l) KiB"
check "1: defaults (no swap; daemon in an SSH shell)"
sudo fallocate -l 16G $SWAP && sudo chmod 600 $SWAP && sudo mkswap -q $SWAP && sudo swapon $SWAP
printf 'forbid_worker_swap = false\n' > $H/config.toml
check "2: swap on, user-scope backstop off, 8 MiB lock limit"

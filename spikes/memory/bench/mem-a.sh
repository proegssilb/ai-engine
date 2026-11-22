#!/usr/bin/env bash
# Memory spike, phase A: no memory pressure. Experts (16.7 GB) all in the
# RAM tier; attention on the two GPUs. One process per variant.
set -u
BIN=/srv/ollama/spike/mem-target/release/memory-spike
M=/srv/ollama/spike/models/Qwen3-30B-A3B-Q4_K_M.gguf
OUT=/srv/ollama/spike/memory-results; mkdir -p $OUT
COMMON="--threads 8 --prompt 512 --tokens 64 --repeat 3"
UNLIMITED="sudo prlimit --memlock=unlimited:unlimited setpriv --reuid=minifig --regid=minifig --init-groups"
run() { local label=$1; shift
  echo "== $label"; sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  timeout 900 "$@" $COMMON --label $label > $OUT/$label.out 2> $OUT/$label.err; echo "exit=$?"
  grep '^RESULT' $OUT/$label.out | sed 's/^RESULT //' | python3 -c 'import sys,json
for l in sys.stdin:
  r=json.loads(l); b=r["buffers_after_runs"]; big=max(b,key=lambda x:x["mib"]) if b else {}
  s=big.get("smaps",{})
  print(f"  load {r[\"load_s\"]:.1f}s  prefill {r[\"prefill_tps\"]:.1f}  decode {r[\"decode_tps\"]:.2f} tok/s | mlock {big.get(\"mlock\")} hugetlb {big.get(\"hugetlb\")} cuda {big.get(\"cuda_register\")} | page {s.get(\"page_kib\")}KiB thp {s.get(\"thp_mib\",0):.0f} hugetlb {s.get(\"hugetlb_mib\",0):.0f} locked {s.get(\"locked_mib\",0):.0f} MiB | VmLck {r[\"VmLck\"]} VmPin {r[\"VmPin\"]}")'
}
ulimit -l
run a1-default-8MiB-limit  $BIN $M                       # as everything ran today: mlock asked, limit 8 MiB
run a2-mlock               $UNLIMITED $BIN $M
run a3-mlock-thp-off       $UNLIMITED $BIN $M --thp off
run a4-nolock-thp-off      $BIN $M --no-mlock --thp off
run a5-cudareg-8MiB-limit  $BIN $M --no-mlock --cuda-register
run a6-mlock-cudareg       $UNLIMITED $BIN $M --cuda-register
# reserved 2 MiB pages: 16.7 GB of experts in several buffers, reserve a little extra
echo 8800 | sudo tee /proc/sys/vm/nr_hugepages >/dev/null; grep HugePages_Total /proc/meminfo
run a7-hugetlb-2M          $UNLIMITED $BIN $M --hugetlb 2M
run a8-hugetlb-2M-cudareg  $UNLIMITED $BIN $M --hugetlb 2M --cuda-register
echo 0 | sudo tee /proc/sys/vm/nr_hugepages >/dev/null
# reserved 1 GiB pages, if the kernel can find them at runtime
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null
echo 20 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages >/dev/null; echo "1G pages: $(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages)"
run a9-hugetlb-1G          $UNLIMITED $BIN $M --hugetlb 1G
echo 0 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages >/dev/null
grep -E "HugePages_Total|Hugetlb" /proc/meminfo
echo PHASE_A_DONE

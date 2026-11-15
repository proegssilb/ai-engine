# Spike: locking and huge pages for the RAM tier

**Questions:**
1. Does locking the RAM tier (`mlock`) work, and what does it cost?
2. Does it matter: under memory pressure, does an unlocked RAM tier get
   swapped out, and what does locking do to the disk tier?
3. Do huge pages (transparent, or reserved 2 MiB / 1 GiB) change speed?
4. What does CUDA registration (`cudaHostRegister`) do to all this?

**Answers, short:** locking works and is free once the lock limit allows it.
That limit is 8 MiB by default on both machines checked, so the RAM tier has
never actually been locked until now. Locking does what the RAM tier
promises: under pressure, unlocked RAM-tier weights *were* swapped out (2.4 of
8.4 GB) and locked ones weren't. But protecting them pushed eviction onto the
disk tier, and overall speed was *lower* with locking under pressure. CUDA
registration locks pages too, even under the 8 MiB limit. Huge pages of any
kind made no measurable difference to inference speed.

## Setup

GPU box (Debian 13 VM, 2× V100-class 32 GB, Ryzen 5800X 8 vCPU, 47 GB RAM,
kernel 6.12, transparent huge pages `always`, no swap). Qwen3-30B-A3B Q4_K_M:
attention on both GPUs, routed experts in the RAM tier (all of them in phase A,
half in phase B with the other half on the disk tier). 512-token prompt,
64 generated tokens, median of 3 runs, page cache dropped before each run.
`src/main.rs` reports per RAM-tier buffer what the kernel did, from
`/proc/self/smaps`: resident, transparent huge pages, reserved huge pages,
locked, swapped.

The lock limit (`RLIMIT_MEMLOCK`) on the box is 8 MiB soft and hard, for the
SSH user and for the ollama service (`LimitMEMLOCK=8388608`); the dev
container is the same. Runs marked "locked" raised it with
`sudo prlimit --memlock=unlimited:unlimited setpriv --reuid=minifig ...`.

## Phase A: no memory pressure (16.7 GB of experts in RAM)

| Variant | Load | Prefill tok/s | Decode tok/s | What the kernel did |
|---|---:|---:|---:|---|
| default (lock asked, 8 MiB limit) | 12.9 s | 231 | 13.9 | `mlock` **failed** (errno 12); THP on all of it |
| locked | 13.3 s | 231 | 13.7 | 16.7 GB locked; THP |
| locked, THP off | 16.2 s | 232 | 14.1 | locked; 4 KiB pages |
| unlocked, THP off | 18.6 s | 232 | 14.1 | 4 KiB pages |
| CUDA-registered, 8 MiB limit | 13.8 s | 238 | 13.9 | registration succeeded despite the limit |
| locked + CUDA-registered | 13.9 s | 238 | 13.8 | |
| reserved 2 MiB pages | 12.9 s | 231 | 13.9 | all on 2 MiB pages |
| reserved 2 MiB + CUDA-registered | 13.6 s | 238 | 14.0 | registration works on them |
| reserved 1 GiB pages | 13.2 s | 232 | 14.0 | kernel found 0 of 20 asked at runtime; fell back to normal pages |

- **Locking costs nothing measurable at load** (+0.4 s on 16.7 GB, within noise).
- **Huge pages don't change inference speed** (13.7–14.1 tok/s decode across
  every variant). Turning THP off only makes loading slower (faulting 4 KiB
  pages). The RAM tier's expert computation isn't TLB-bound here.
- **1 GiB pages need a boot-time reservation**: a running machine with a busy
  page cache can't assemble them, even after dropping caches and compacting.
- **CUDA registration adds ~3% to prompt processing** (op offload copies from
  page-locked memory), as in the earlier pinned-vs-pageable A/B.
- **`VmPin` stayed 0** with CUDA registration, so the driver's pinning isn't
  visible in that counter; phase B shows what it does.

## Phase B: memory pressure (8.4 GB experts in RAM, 8.4 GB on disk, 12 GB cap)

A temporary 32 GB swap file; the process capped at 12 GB with
`systemd-run --scope -p MemoryMax=12G`, so the RAM tier, the disk tier's page
cache and everything else compete. Swap file and reservations removed after.

| Variant | Load | Prefill tok/s | Decode tok/s | RAM tier swapped out |
|---|---:|---:|---:|---:|
| reference (no cap, locked) | 11.1 s | 231 | 13.9 | 0 |
| **unlocked**, 12 GB cap | 24.4 s | 151 | 12.8 | **2,372 MiB** of 8,370 |
| **locked**, 12 GB cap | 22.5 s | 81 | 10.5 | 0 |
| CUDA-registered, unlocked, 12 GB cap | 23.4 s | 79 | 10.2 | 0 |
| reserved 2 MiB pages, unlocked, 12 GB cap | 20.2 s | 230 | 14.2 | 0 |

- **An unlocked RAM tier does get swapped out** under pressure, as feared:
  28% of it here. "RAM tier" quietly becomes partly a swap tier.
- **Locking keeps it in RAM, but costs the disk tier.** With 8.4 GB locked
  inside 12 GB, the disk tier's page cache had less room and was re-read from
  disk more: prompt processing halved (81 vs 151 tok/s) and generation fell to
  10.5 tok/s, *below* the unlocked run. The kernel's own balancing (swap out
  cold RAM-tier experts, keep hot disk-tier pages) was faster overall.
- **CUDA registration locks pages too**, with the 8 MiB limit in force: same
  outcome as `mlock`. On CUDA machines it's a second way to lock the RAM tier,
  and one that doesn't depend on the lock limit.
- **Reserved huge pages are never swapped and aren't charged to the memory
  cap** (they belong to the separate hugetlb controller), so that run had
  effectively 12 GB *plus* the RAM tier. Fastest, but not comparable: it
  escaped the cap rather than living within it.

## What this means for the product

1. **Locking is the right meaning of "RAM tier", and it needs the limit
   raised.** It's the only way the placement users chose holds under pressure,
   and it's free to do. The 8 MiB default defeats it silently; the worker's
   warning (added today) and `LimitMEMLOCK=infinity` in the systemd unit are
   necessary. Worth considering: register the RAM tier with CUDA by default on
   CUDA machines, which locks it regardless of the limit and speeds prompt
   processing ~3%.
2. **Locked RAM comes out of everyone else's memory.** Under pressure it moves
   eviction onto the disk tier, which can be slower overall. So the fit check
   should leave the disk tier page-cache headroom (at least its hot share)
   rather than counting locked RAM against free memory alone, and the UI
   should say that a disk tier under a tight memory budget will thrash.
3. **Huge pages aren't worth exposing.** No speed difference; transparent huge
   pages (the default here) cost nothing. Reserved pages are an operator
   option at most: they need a reservation, 1 GiB pages need it at boot, and
   they sit outside memory-cap accounting.
4. **The earlier GPU spike's "pinned RAM" was never locked** (8 MiB limit),
   but the box has no swap, so those results stand.

## Run it

```bash
cd spikes/memory && cargo build --release --features cuda
target/release/memory-spike MODEL.gguf [--ram-share F] [--no-mlock] [--thp on|off] \
    [--hugetlb 2M|1G] [--cuda-register] [--threads N] [--prompt N] [--tokens N] [--repeat N]
```

The phase scripts used on the box are in `bench/`.

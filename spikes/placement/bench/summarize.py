#!/usr/bin/env python3
"""Turn a test-box results directory into one markdown table."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []

for f in sorted(out.glob("ref-*.jsonl")):
    pp = tg = None
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("n_prompt"):
            pp = r["avg_ts"]
        else:
            tg = r["avg_ts"]
    rows.append((f.stem, pp, tg, None, "llama-bench reference", ""))

res = out / "results.jsonl"
for line in res.read_text().splitlines() if res.exists() else []:
    r = json.loads(line)
    homes = ", ".join(
        f"{h} {v['mib'] / 1024:.1f}G" + ("" if v["in_ram"] is None else f" ({v['in_ram']:.0%} in RAM)")
        for h, v in r["homes"].items()
    )
    cold = r.get("cold_decode_tps")
    rows.append((r["label"], r["prefill_tps"], r["decode_tps"], cold, homes, r["logits_fnv"][:8]))

def n(x):
    return "-" if x is None else f"{x:.1f}"

print("| run | prefill tok/s | decode tok/s | decode after evict | where the weights are | logits |")
print("|---|---:|---:|---:|---|---|")
for label, pp, tg, cold, homes, fnv in rows:
    print(f"| {label} | {n(pp)} | {n(tg)} | {n(cold)} | {homes} | {fnv} |")
print("\nSame `logits` hash = bit-identical output. Different kernels (GPU vs CPU, repacked vs not)"
      " legitimately change the hash; same kernels in different memory must not.")

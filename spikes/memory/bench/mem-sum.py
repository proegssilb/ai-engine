import json, sys, glob, os
for path in sorted(glob.glob(sys.argv[1] + "/*.out")):
    label = os.path.basename(path)[:-4]
    lines = [l[7:] for l in open(path) if l.startswith("RESULT ")]
    if not lines:
        err = open(path[:-4] + ".err").read().strip().splitlines()[-2:]
        print(f"{label:24} NO RESULT: {' | '.join(err)}")
        continue
    r = json.loads(lines[0])
    bufs = r["buffers_after_runs"]
    tot = lambda k: sum(b["smaps"].get(k, 0) for b in bufs)
    mib = sum(b["mib"] for b in bufs)
    first = bufs[0] if bufs else {}
    print(f"{label:24} load {r['load_s']:5.1f}s  prefill {r['prefill_tps']:6.1f}  decode {r['decode_tps']:5.2f} tok/s  "
          f"| {len(bufs)} buf {mib:6.0f} MiB: page {max((b['smaps']['page_kib'] for b in bufs), default=0)}KiB "
          f"thp {tot('thp_mib'):5.0f} hugetlb {tot('hugetlb_mib'):5.0f} locked {tot('locked_mib'):5.0f} swap {tot('swap_mib'):5.0f} "
          f"| mlock {first.get('mlock')} hugetlb {first.get('hugetlb')} cudareg {first.get('cuda_register')} "
          f"| VmLck {r['VmLck']} VmPin {r.get('VmPin','?')} VmSwap {r['VmSwap']}")

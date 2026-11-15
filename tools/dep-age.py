#!/usr/bin/env python3
"""Report the age of every crates.io dependency in Cargo.lock.

Lists each locked crate version with its publish time from the crates.io
sparse index, and flags those younger than the threshold (default 14 days).
Report only; enforcement, if wanted, is Cargo's own
`[registry] global-min-publish-age` setting (Cargo 1.100+).

Usage: tools/dep-age.py [--days N] [--all] [path/to/Cargo.lock]
"""
import argparse, datetime as dt, json, sys, tomllib, urllib.request
from concurrent.futures import ThreadPoolExecutor

def index_path(name):
    n = name.lower()
    if len(n) <= 2:
        return f"{len(n)}/{n}"
    if len(n) == 3:
        return f"3/{n[0]}/{n}"
    return f"{n[:2]}/{n[2:4]}/{n}"

def pubtime(name, version):
    req = urllib.request.Request(f"https://index.crates.io/{index_path(name)}", headers={"User-Agent": "dep-age-report"})
    with urllib.request.urlopen(req, timeout=30) as r:
        for line in r.read().decode().splitlines():
            d = json.loads(line)
            if d["vers"] == version:
                return d.get("pubtime")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lockfile", nargs="?", default="Cargo.lock")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--all", action="store_true", help="list every crate, not just the young ones")
    a = ap.parse_args()
    lock = tomllib.load(open(a.lockfile, "rb"))
    pkgs = [(p["name"], p["version"]) for p in lock.get("package", []) if p.get("source", "").startswith("registry+")]
    with ThreadPoolExecutor(16) as ex:
        times = list(ex.map(lambda p: pubtime(*p), pkgs))
    now = dt.datetime.now(dt.timezone.utc)
    rows, young = [], 0
    for (name, ver), t in sorted(zip(pkgs, times), key=lambda x: x[1] or ""):
        if t is None:
            rows.append((name, ver, "unknown", None)); continue
        age = (now - dt.datetime.fromisoformat(t.replace("Z", "+00:00"))).days
        young += age < a.days
        rows.append((name, ver, t[:10], age))
    for name, ver, t, age in rows:
        if a.all or age is None or age < a.days:
            flag = "  < %dd" % a.days if age is not None and age < a.days else ""
            print(f"{name:32} {ver:14} {t:12} {'' if age is None else str(age)+'d':>6}{flag}")
    print(f"{len(pkgs)} crates, {young} younger than {a.days} days", file=sys.stderr)
    return 0

sys.exit(main())

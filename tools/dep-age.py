#!/usr/bin/env python3
"""Report the age of every dependency in a Cargo.lock or package-lock.json.

Lists each locked crate version with its publish time from the crates.io
sparse index, and flags those younger than the threshold (default 14 days).
Report only; enforcement, if wanted, is Cargo's own
`[registry] global-min-publish-age` setting (Cargo 1.100+).

Usage: tools/dep-age.py [--days N] [--all] [Cargo.lock | ui/package-lock.json]
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

def npm_pubtime(name, version, cache={}):
    if name not in cache:
        req = urllib.request.Request(f"https://registry.npmjs.org/{name}", headers={"User-Agent": "dep-age-report"})
        with urllib.request.urlopen(req, timeout=30) as r:
            cache[name] = json.load(r).get("time", {})
    return cache[name].get(version)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lockfile", nargs="?", default="Cargo.lock")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--all", action="store_true", help="list every crate, not just the young ones")
    a = ap.parse_args()
    if a.lockfile.endswith("package-lock.json"):
        lock = json.load(open(a.lockfile))
        pkgs = sorted({(path.split("node_modules/")[-1], p["version"]) for path, p in lock.get("packages", {}).items() if path and "version" in p})
        lookup = npm_pubtime
    else:
        lock = tomllib.load(open(a.lockfile, "rb"))
        pkgs = [(p["name"], p["version"]) for p in lock.get("package", []) if p.get("source", "").startswith("registry+")]
        lookup = pubtime
    with ThreadPoolExecutor(16) as ex:
        times = list(ex.map(lambda p: lookup(*p), pkgs))
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
    print(f"{len(pkgs)} packages, {young} younger than {a.days} days", file=sys.stderr)
    return 0

sys.exit(main())

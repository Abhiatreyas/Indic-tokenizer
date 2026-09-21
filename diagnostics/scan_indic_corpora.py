"""Discover which languages the local Indic corpora contain, and where.

indiccorp_v2 / sangraha are multi-language JSONL with a "group" field. The first
200k lines were all Kannada, so the layout has to be discovered rather than
assumed. Reads sequentially and reports the group seen at each checkpoint; stops
once a line cap is reached or the file ends.

    python diagnostics/scan_indic_corpora.py --cap-lines 6000000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List

# Cheap substring extraction; full json.loads on millions of lines is wasteful
# when only one field is needed.
GROUP_RE = re.compile(r'"group"\s*:\s*"([^"]+)"')
TEXT_RE = re.compile(r'"text"\s*:\s*"')


def scan(path: str, cap_lines: int, report_every: int) -> Dict[str, dict]:
    seen: Dict[str, dict] = {}
    order: List[str] = []
    t0 = time.time()
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            m = GROUP_RE.search(line)
            if m:
                g = m.group(1)
                if g not in seen:
                    seen[g] = {"first_line": n, "count": 0}
                    order.append(g)
                    print(f"    NEW group={g!r} at line {n:,} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                seen[g]["count"] += 1
            if n % report_every == 0:
                print(f"    line {n:,}  groups={len(seen)}  "
                      f"{time.time()-t0:.0f}s", flush=True)
            if n >= cap_lines:
                break
    return seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", default=[
        "G:/jnana_raw/indiccorp_v2.jsonl",
        "G:/jnana_raw/sangraha_verified.jsonl",
    ])
    ap.add_argument("--cap-lines", type=int, default=6_000_000)
    ap.add_argument("--report-every", type=int, default=1_000_000)
    args = ap.parse_args(argv)

    summary = {}
    for path in args.files:
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        size_gb = os.path.getsize(path) / 1e9
        print(f"\n=== {path}  ({size_gb:.1f} GB) ===")
        seen = scan(path, args.cap_lines, args.report_every)
        summary[path] = seen
        print(f"  groups found: {len(seen)}")
        for g, info in sorted(seen.items(), key=lambda kv: kv[1]["first_line"]):
            print(f"    {g:<24} first_line={info['first_line']:>12,} "
                  f"count={info['count']:>9,}")

    with open("artifacts/indic_scan.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwritten: artifacts/indic_scan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

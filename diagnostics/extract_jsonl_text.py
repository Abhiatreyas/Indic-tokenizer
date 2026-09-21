"""Extract a text sample from a huge JSONL corpus without loading it into memory.

Reads sequentially and stops once the character target is reached, so a 130 GB
file costs only the bytes it actually needs.

    python diagnostics/extract_jsonl_text.py \
        --source "G:/jnana_raw/culturax_kn.jsonl" \
        --out data_real/kn_train.jsonl --target-chars 100000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-chars", type=int, default=50_000_000)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--max-doc-chars", type=int, default=40_000)
    ap.add_argument("--min-doc-chars", type=int, default=100)
    ap.add_argument("--skip-lines", type=int, default=0)
    args = ap.parse_args(argv)

    if not os.path.exists(args.source):
        print(f"missing source: {args.source}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = 0
    docs = 0
    seen = 0
    bad = 0
    with open(args.source, "r", encoding="utf-8", errors="replace") as fh, open(
        args.out, "w", encoding="utf-8"
    ) as out:
        for line in fh:
            if seen < args.skip_lines:
                seen += 1
                continue
            if total >= args.target_chars:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            text = rec.get(args.text_field)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < args.min_doc_chars:
                continue
            text = text[: args.max_doc_chars]
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total += len(text)
            docs += 1
            if docs % 20_000 == 0:
                print(f"    {docs} docs, {total} chars", flush=True)

    print(
        f"  wrote {args.out}: {docs} docs, {total} chars "
        f"(bad_json={bad}, target={args.target_chars})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

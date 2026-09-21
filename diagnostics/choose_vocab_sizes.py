"""Choose slot vocabulary sizes by marginal gain and utilization, not fertility.

Fertility falls monotonically with vocabulary size, so "minimise fertility" has no
interior optimum -- it just says "bigger". The sweep's knee column degenerated to
the largest size tested for exactly that reason.

Two criteria that DO have an optimum:

  * marginal gain  -- percent fertility bought per doubling of rows. Once this
                      drops below a few percent, more rows are poor value.
  * utilization    -- fraction of the vocabulary that actually fires on held-out
                      text. A slot whose pieces mostly never fire is oversized,
                      and those rows are dead weight in the embedding table.

    python diagnostics/choose_vocab_sizes.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    UnrepresentableRun,
    train_slot,
)

ML = "data_real/multilang"
RO = "data_real/romanized"

# label -> (script, train paths, held path, vocab sizes to test)
CASES: Dict[str, Tuple[str, List[str], str, Tuple[int, ...]]] = {
    "DEVANAGARI (hi+mr+ne)": ("DEVANAGARI",
        ["data_real/hi_train.jsonl", f"{ML}/mr.jsonl", f"{ML}/ne.jsonl"],
        "data_real/hi_heldout.jsonl", (16000, 32000, 64000, 100000)),
    "LATIN (en+hinglish+tanglish)": ("LATIN",
        ["data_real/en_train.jsonl", f"{RO}/hinglish.jsonl", f"{RO}/tanglish.jsonl"],
        "data_real/en_heldout.jsonl", (16000, 32000, 64000, 100000)),
    "KANNADA (kn)": ("KANNADA", ["data_real/kn_train.jsonl"],
        "data_real/kn_heldout.jsonl", (16000, 32000, 64000)),
    "TAMIL (ta)": ("TAMIL", [f"{ML}/ta.jsonl"], f"{ML}/ta.jsonl",
        (8000, 16000, 32000, 64000)),
    "MALAYALAM (ml)": ("MALAYALAM", [f"{ML}/ml.jsonl"], f"{ML}/ml.jsonl",
        (8000, 16000, 32000, 64000)),
    "ARABIC/Urdu (ur)": ("ARABIC", [f"{ML}/ur.jsonl"], f"{ML}/ur.jsonl",
        (8000, 16000, 32000, 64000)),
}


def load_docs(path: str, cap: int) -> List[str]:
    out, total = [], 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)["text"]
            out.append(t)
            total += len(t)
            if total >= cap:
                break
    return out


def runs_for(docs: Sequence[str], script: str) -> List[str]:
    from tokenizer.scripts import CODE_MATH, COMMON, LATIN, NEUTRAL, segment

    slot_map = {COMMON: NEUTRAL, LATIN: LATIN, script: script}
    out = []
    for t in docs:
        for r in segment(t, slot_map=slot_map, structural_slot=CODE_MATH,
                         whitespace_ownership="leading"):
            if r.slot == script:
                out.append(r.text)
    return out


def encode_ids(slot, runs: Sequence[str]) -> List[int]:
    """All local ids, with per-character fallback counted as byte ids."""
    ids: List[int] = []
    for text in runs:
        stack = [text]
        while stack:
            c = stack.pop()
            if not c:
                continue
            try:
                ids.extend(slot.encode(c))
                continue
            except UnrepresentableRun:
                pass
            if len(c) == 1:
                ids.extend(c.encode("utf-8"))
                continue
            mid = len(c) // 2
            stack.append(c[mid:])
            stack.append(c[:mid])
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-chars", type=int, default=1_400_000)
    ap.add_argument("--held-chars", type=int, default=350_000)
    ap.add_argument("--out", default="artifacts/vocab_choice.json")
    args = ap.parse_args(argv)

    rows: List[dict] = []
    for label, (script, train_paths, held_path, vocabs) in CASES.items():
        per = max(1, args.train_chars // max(1, len(train_paths)))
        train_runs: List[str] = []
        for p in train_paths:
            if os.path.exists(p):
                train_runs += runs_for(load_docs(p, per), script)
        held_runs = runs_for(load_docs(held_path, args.held_chars), script) \
            if os.path.exists(held_path) else []
        if len(train_runs) < 100 or len(held_runs) < 50:
            print(f"  {label}: insufficient data, skipping")
            continue
        words = sum(len(r.split()) for r in held_runs)
        print(f"\n{label}: train={len(train_runs)} runs "
              f"({sum(len(r) for r in train_runs)} ch), held={words} words")

        prev_tpw = None
        for vocab in vocabs:
            t0 = time.time()
            slot = train_slot(script, "unigram", list(train_runs), vocab,
                              alphabet_scope=ALPHABET_OBSERVED_ASCII,
                              max_piece_length=48)
            ids = encode_ids(slot, held_runs)
            distinct = len(set(ids))
            tpw = len(ids) / words
            util = distinct / max(1, slot.local_size)
            gain = None if prev_tpw is None else (prev_tpw - tpw) / prev_tpw * 100
            rows.append({
                "condition": label, "script": script, "vocab": vocab,
                "pieces": slot.local_size, "distinct_fired": distinct,
                "utilization": round(util, 4),
                "tokens_per_word": round(tpw, 4),
                "gain_pct_vs_prev": None if gain is None else round(gain, 2),
                "sec": round(time.time() - t0, 1),
            })
            g = "  -  " if gain is None else f"{gain:+.1f}%"
            print(f"   vocab={vocab:<7} pieces={slot.local_size:<7} "
                  f"fired={distinct:<7} util={util:.3f}  tok/word={tpw:.4f}  "
                  f"gain={g}  ({time.time()-t0:.0f}s)")
            prev_tpw = tpw

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== marginal gain and utilization ===")
    print(f"{'condition':<32}{'vocab':>8}{'util':>8}{'tok/word':>10}{'gain':>9}")
    for r in rows:
        g = "-" if r["gain_pct_vs_prev"] is None else f"{r['gain_pct_vs_prev']:+.1f}%"
        print(f"{r['condition']:<32}{r['vocab']:>8}{r['utilization']:>8.3f}"
              f"{r['tokens_per_word']:>10.4f}{g:>9}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

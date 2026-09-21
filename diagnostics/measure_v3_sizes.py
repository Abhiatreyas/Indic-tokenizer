"""Measure the cost of the proposed v3 vocabulary reductions.

Proposed: Kannada, Telugu, Tamil, Malayalam -> 32k; Oriya and Punjabi -> 16k;
everything else unchanged. This measures each changed slot at its old and new
size on the same held-out runs so the cost is observed rather than interpolated.

    python diagnostics/measure_v3_sizes.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    UnrepresentableRun,
    train_slot,
)

ML = "data_real/multilang"

# script -> (name, old vocab, new vocab, train paths, held path)
CHANGES: Dict[str, Tuple[str, int, int, List[str], str]] = {
    "KANNADA": ("Kannada", 48000, 32000, ["data_real/kn_train.jsonl"],
                "data_real/kn_heldout.jsonl"),
    "MALAYALAM": ("Malayalam", 48000, 32000, [f"{ML}/ml.jsonl"], f"{ML}/ml.jsonl"),
    "TELUGU": ("Telugu", 48000, 32000, [f"{ML}/te.jsonl"], f"{ML}/te.jsonl"),
    "TAMIL": ("Tamil", 40000, 32000, [f"{ML}/ta.jsonl"], f"{ML}/ta.jsonl"),
    "ORIYA": ("Odia", 24000, 16000, [f"{ML}/or.jsonl"], f"{ML}/or.jsonl"),
    "GURMUKHI": ("Punjabi", 20000, 16000, [f"{ML}/pa.jsonl"], f"{ML}/pa.jsonl"),
}

# Kept as-is, measured for reference so the table is complete.
UNCHANGED: Dict[str, Tuple[str, int, List[str], str]] = {
    "LATIN": ("English+Hinglish+Tanglish", 64000,
              ["data_real/en_train.jsonl", "data_real/romanized/hinglish.jsonl",
               "data_real/romanized/tanglish.jsonl"], "data_real/en_heldout.jsonl"),
    "DEVANAGARI": ("Hindi+Marathi", 48000,
                   ["data_real/hi_train.jsonl", f"{ML}/mr.jsonl"],
                   "data_real/hi_heldout.jsonl"),
    "BENGALI": ("Bengali+Assamese", 32000, [f"{ML}/bn.jsonl"], f"{ML}/bn.jsonl"),
    "GUJARATI": ("Gujarati", 24000, [f"{ML}/gu.jsonl"], f"{ML}/gu.jsonl"),
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


def count(slot, runs: Sequence[str]) -> Tuple[int, int]:
    total, fired = 0, set()
    for text in runs:
        stack = [text]
        while stack:
            c = stack.pop()
            if not c:
                continue
            try:
                ids = slot.encode(c)
                total += len(ids)
                fired.update(ids)
                continue
            except UnrepresentableRun:
                pass
            if len(c) == 1:
                bs = list(c.encode("utf-8"))
                total += len(bs)
                fired.update(bs)
                continue
            mid = len(c) // 2
            stack.append(c[mid:])
            stack.append(c[:mid])
    return total, len(fired)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-chars", type=int, default=1_400_000)
    ap.add_argument("--held-chars", type=int, default=350_000)
    ap.add_argument("--out", default="artifacts/v3_sizes.json")
    args = ap.parse_args(argv)

    rows: List[dict] = []

    def trial(label, script, vocab, paths, held, changed):
        per = max(1, args.train_chars // max(1, len(paths)))
        tr: List[str] = []
        for p in paths:
            if os.path.exists(p):
                tr += runs_for(load_docs(p, per), script)
        he = runs_for(load_docs(held, args.held_chars), script)
        if len(tr) < 100 or len(he) < 50:
            print(f"  {label}: insufficient data")
            return None
        words = sum(len(r.split()) for r in he)
        t0 = time.time()
        slot = train_slot(script, "unigram", list(tr), vocab,
                          alphabet_scope=ALPHABET_OBSERVED_ASCII,
                          max_piece_length=48)
        toks, fired = count(slot, he)
        rec = {"label": label, "script": script, "vocab": vocab,
               "changed": changed, "pieces": slot.local_size,
               "fired": fired, "utilization": round(fired / max(1, slot.local_size), 3),
               "tokens_per_word": round(toks / words, 4),
               "sec": round(time.time() - t0, 1)}
        rows.append(rec)
        print(f"   {label:<26} vocab={vocab:<7} pieces={slot.local_size:<7} "
              f"util={rec['utilization']:<6} tok/word={toks/words:.4f} "
              f"({time.time()-t0:.0f}s)")
        return rec

    print("=== CHANGED slots: old vs new size ===")
    for script, (name, old, new, paths, held) in CHANGES.items():
        print(f"\n {name} ({script})")
        trial(f"{name} OLD {old//1000}k", script, old, paths, held, True)
        trial(f"{name} NEW {new//1000}k", script, new, paths, held, True)

    print("\n=== UNCHANGED slots (reference) ===")
    for script, (name, vocab, paths, held) in UNCHANGED.items():
        print(f"\n {name} ({script})")
        trial(f"{name} {vocab//1000}k", script, vocab, paths, held, False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== cost of each reduction ===")
    print(f"{'slot':<14}{'old':>7}{'new':>7}{'old t/w':>10}{'new t/w':>10}"
          f"{'cost':>9}{'util old':>10}{'util new':>10}")
    by = {}
    for r in rows:
        by.setdefault(r["script"], []).append(r)
    total_delta = 0.0
    for script, (name, old, new, _, _) in CHANGES.items():
        rs = sorted(by.get(script, []), key=lambda x: -x["vocab"])
        if len(rs) < 2:
            continue
        o, n = rs[0], rs[1]
        cost = (n["tokens_per_word"] - o["tokens_per_word"]) / o["tokens_per_word"] * 100
        total_delta += n["tokens_per_word"] - o["tokens_per_word"]
        print(f"{name:<14}{old:>7}{new:>7}{o['tokens_per_word']:>10.4f}"
              f"{n['tokens_per_word']:>10.4f}{cost:>8.1f}%"
              f"{o['utilization']:>10.3f}{n['utilization']:>10.3f}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

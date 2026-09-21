"""Does dropping languages from a shared slot actually shrink its vocabulary?

The proposed change removes Nepali and Konkani from DEVANAGARI (leaving Hindi +
Marathi), Tulu from KANNADA, the international languages from LATIN, and Urdu
from ARABIC. Only ARABIC loses its slot outright; the rest keep their script.

This measures the one case where the answer is not obvious: does a 2-language
Devanagari slot need fewer rows than a 3-language one, and does a LATIN slot with
more romanized variants need more?

    python diagnostics/measure_slot_pruning.py
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
RO = "data_real/romanized"

TRAIN = {
    "hi": "data_real/hi_train.jsonl",
    "mr": f"{ML}/mr.jsonl",
    "ne": f"{ML}/ne.jsonl",
}
HELD = {
    "hi": "data_real/hi_heldout.jsonl",
}
RO_TRAIN = {
    "en": "data_real/en_train.jsonl",
    "hinglish": f"{RO}/hinglish.jsonl",
    "tanglish": f"{RO}/tanglish.jsonl",
    "tenglish": f"{RO}/tenglish.jsonl",
}
RO_HELD = {"en": "data_real/en_heldout.jsonl"}


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


def eval_tokens(slot, runs: Sequence[str]) -> int:
    total = 0
    for text in runs:
        stack = [text]
        while stack:
            c = stack.pop()
            if not c:
                continue
            try:
                total += len(slot.encode(c))
                continue
            except UnrepresentableRun:
                pass
            if len(c) == 1:
                total += len(c.encode("utf-8"))
                continue
            mid = len(c) // 2
            stack.append(c[mid:])
            stack.append(c[:mid])
    return total


def trial(label, script, train_paths, held_path, vocab, char_cap, rows):
    per = max(1, char_cap // max(1, len(train_paths)))
    train_runs: List[str] = []
    for p in train_paths:
        if os.path.exists(p):
            train_runs += runs_for(load_docs(p, per), script)
    held_runs = runs_for(load_docs(held_path, char_cap // 4), script)
    if len(train_runs) < 100 or len(held_runs) < 50:
        print(f"  {label}: insufficient data")
        return None
    words = sum(len(r.split()) for r in held_runs)
    t0 = time.time()
    slot = train_slot(script, "unigram", list(train_runs), vocab,
                      alphabet_scope=ALPHABET_OBSERVED_ASCII,
                      max_piece_length=48)
    toks = eval_tokens(slot, held_runs)
    rec = {"label": label, "script": script, "vocab": vocab,
           "languages": len(train_paths), "train_runs": len(train_runs),
           "train_chars": sum(len(r) for r in train_runs),
           "tokens_per_word": round(toks / words, 4),
           "sec": round(time.time() - t0, 1)}
    rows.append(rec)
    print(f"   {label:<44} langs={len(train_paths)} "
          f"train={rec['train_chars']:<9} vocab={vocab:<6} "
          f"tok/word={toks/words:.4f}  ({time.time()-t0:.0f}s)")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--char-cap", type=int, default=1_400_000)
    ap.add_argument("--out", default="artifacts/slot_pruning.json")
    args = ap.parse_args(argv)
    rows: List[dict] = []

    print("=== DEVANAGARI: does dropping Nepali shrink the slot? ===")
    for vocab in (32000, 64000):
        trial("DEVANAGARI hi+mr+ne (current)", "DEVANAGARI",
              [TRAIN["hi"], TRAIN["mr"], TRAIN["ne"]], HELD["hi"], vocab,
              args.char_cap, rows)
        trial("DEVANAGARI hi+mr (after dropping ne)", "DEVANAGARI",
              [TRAIN["hi"], TRAIN["mr"]], HELD["hi"], vocab,
              args.char_cap, rows)

    print("\n=== LATIN: does adding romanized variants cost more rows? ===")
    if os.path.exists(RO_TRAIN["tenglish"]):
        for vocab in (32000, 64000):
            trial("LATIN en+hinglish+tanglish (current)", "LATIN",
                  [RO_TRAIN["en"], RO_TRAIN["hinglish"], RO_TRAIN["tanglish"]],
                  RO_HELD["en"], vocab, args.char_cap, rows)
            trial("LATIN en+hinglish+tanglish+tenglish", "LATIN",
                  [RO_TRAIN["en"], RO_TRAIN["hinglish"], RO_TRAIN["tanglish"],
                   RO_TRAIN["tenglish"]], RO_HELD["en"], vocab,
                  args.char_cap, rows)
    else:
        print(f"  no tenglish corpus at {RO_TRAIN['tenglish']} (skipping this half)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== summary: held-out tokens/word ===")
    for r in rows:
        print(f"  {r['label']:<44} vocab={r['vocab']:<6} "
              f"langs={r['languages']} tok/word={r['tokens_per_word']}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

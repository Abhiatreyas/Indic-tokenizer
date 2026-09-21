"""Fertility vs vocabulary size, per slot, with the recommended algorithm.

The algorithm decision is settled (Unigram, max_piece_length=48, except NEUTRAL
and CODE_MATH which stay BPE). What is not yet measured is how big each slot's
vocabulary should be, so this sweeps it and finds the knee: the smallest size
within 2% of the best observed, i.e. the point past which extra embedding rows buy
almost nothing.

One representative language per script, plus the two SHARED slots, because those
are the configurations the design actually uses.

    python diagnostics/sweep_vocab_sizes.py
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

# label -> (kind, script, [(path, is_train), ...])
CONDITIONS: Dict[str, Tuple[str, str, List[Tuple[str, bool]]]] = {
    "DEVANAGARI (shared: hi+mr+ne)": ("shared", "DEVANAGARI", [
        ("data_real/hi_train.jsonl", True), (f"{ML}/mr.jsonl", True),
        (f"{ML}/ne.jsonl", True)]),
    "LATIN (shared: en+hinglish+tanglish)": ("shared", "LATIN", [
        ("data_real/en_train.jsonl", True), (f"{RO}/hinglish.jsonl", True),
        (f"{RO}/tanglish.jsonl", True)]),
    "KANNADA (kn)": ("single", "KANNADA", [("data_real/kn_train.jsonl", True)]),
    "BENGALI (bn)": ("single", "BENGALI", [(f"{ML}/bn.jsonl", True)]),
    "TAMIL (ta)": ("single", "TAMIL", [(f"{ML}/ta.jsonl", True)]),
    "TELUGU (te)": ("single", "TELUGU", [(f"{ML}/te.jsonl", True)]),
    "MALAYALAM (ml)": ("single", "MALAYALAM", [(f"{ML}/ml.jsonl", True)]),
    "GUJARATI (gu)": ("single", "GUJARATI", [(f"{ML}/gu.jsonl", True)]),
    "GURMUKHI (pa)": ("single", "GURMUKHI", [(f"{ML}/pa.jsonl", True)]),
    "ORIYA (or)": ("single", "ORIYA", [(f"{ML}/or.jsonl", True)]),
    "ARABIC (ur)": ("single", "ARABIC", [(f"{ML}/ur.jsonl", True)]),
}

HELD = {
    "DEVANAGARI (shared: hi+mr+ne)": ("DEVANAGARI", ["data_real/hi_heldout.jsonl"]),
    "LATIN (shared: en+hinglish+tanglish)": ("LATIN", ["data_real/en_heldout.jsonl"]),
    "KANNADA (kn)": ("KANNADA", ["data_real/kn_heldout.jsonl"]),
    "BENGALI (bn)": ("BENGALI", [f"{ML}/bn.jsonl"]),
    "TAMIL (ta)": ("TAMIL", [f"{ML}/ta.jsonl"]),
    "TELUGU (te)": ("TELUGU", [f"{ML}/te.jsonl"]),
    "MALAYALAM (ml)": ("MALAYALAM", [f"{ML}/ml.jsonl"]),
    "GUJARATI (gu)": ("GUJARATI", [f"{ML}/gu.jsonl"]),
    "GURMUKHI (pa)": ("GURMUKHI", [f"{ML}/pa.jsonl"]),
    "ORIYA (or)": ("ORIYA", [f"{ML}/or.jsonl"]),
    "ARABIC (ur)": ("ARABIC", [f"{ML}/ur.jsonl"]),
}

# Big slots get a wider sweep; single scripts need only enough to find the knee.
VOCABS_BIG = (8000, 16000, 32000, 64000)
VOCABS_SMALL = (4000, 8000, 16000, 32000)


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
    from tokenizer.scripts import CODE_MATH, DEFAULT_SLOT_MAP, LATIN, NEUTRAL, segment
    from tokenizer.scripts import COMMON

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-chars", type=int, default=1_400_000)
    ap.add_argument("--held-chars", type=int, default=350_000)
    ap.add_argument("--out", default="artifacts/vocab_sweep.json")
    args = ap.parse_args(argv)

    rows: List[dict] = []
    for label, (kind, script, sources) in CONDITIONS.items():
        _, held_paths = HELD[label]
        train_runs: List[str] = []
        per_source = max(1, args.train_chars // max(1, len(sources)))
        for path, _ in sources:
            if not os.path.exists(path):
                print(f"  {label}: missing {path}")
                continue
            train_runs += runs_for(load_docs(path, per_source), script)
        held_runs: List[str] = []
        for path in held_paths:
            if os.path.exists(path):
                held_runs += runs_for(load_docs(path, args.held_chars), script)
        if len(train_runs) < 100 or len(held_runs) < 50:
            print(f"  {label}: too few runs, skipping")
            continue

        words = sum(len(r.split()) for r in held_runs)
        vocabs = VOCABS_BIG if kind == "shared" or script == "KANNADA" else VOCABS_SMALL
        print(f"\n{label}: train_runs={len(train_runs)} "
              f"({sum(len(r) for r in train_runs)} ch), held={words} words")
        for vocab in vocabs:
            t0 = time.time()
            slot = train_slot(script, "unigram", list(train_runs), vocab,
                              alphabet_scope=ALPHABET_OBSERVED_ASCII,
                              max_piece_length=48)
            toks = eval_tokens(slot, held_runs)
            rows.append({"condition": label, "kind": kind, "script": script,
                         "vocab": vocab, "pieces": slot.local_size,
                         "tokens_per_word": round(toks / words, 4),
                         "sec": round(time.time() - t0, 1)})
            print(f"   vocab={vocab:<6} pieces={slot.local_size:<6} "
                  f"tok/word={toks/words:.4f}  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== fertility vs vocabulary size (tokens/word) ===")
    by_cond: Dict[str, List[dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    all_v = sorted({r["vocab"] for r in rows})
    print(f"{'condition':<38}" + "".join(f"{v:>10}" for v in all_v) + f"{'knee':>10}")
    for cond, rs in by_cond.items():
        m = {r["vocab"]: r["tokens_per_word"] for r in rs}
        best = min(m.values())
        knee = min(v for v in m if m[v] <= best * 1.02)
        row = f"{cond:<38}"
        for v in all_v:
            row += f"{m[v]:>10.4f}" if v in m else f"{'-':>10}"
        print(row + f"{knee:>10}")
    print("\n  knee = smallest vocab within 2% of the best observed")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

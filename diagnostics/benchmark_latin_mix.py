"""The LATIN slot: English + romanized Indic code-mixed (Hinglish, Tanglish).

In this design LATIN is a single shared slot, because the encoder cannot separate
English from romanized Hindi or romanized Kannada -- all three are Latin script.
This measures whether Unigram with a raised piece limit still wins when the slot
holds code-mixed text rather than one clean language, which is different from
every per-script slot in the benchmark.

Held-out is reported per source so a win on English cannot hide a loss on the
romanized mix.

    python diagnostics/benchmark_latin_mix.py --vocab 16000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

from tokenizer.scripts import CODE_MATH, DEFAULT_SLOT_MAP, segment
from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    UnrepresentableRun,
    train_slot,
)

SOURCES = {
    "english": ("data_real/en_train.jsonl", "data_real/en_heldout.jsonl"),
    "hinglish": ("data_real/romanized/hinglish.jsonl",
                 "data_real/romanized/hinglish.jsonl"),
    "tanglish": ("data_real/romanized/tanglish.jsonl",
                 "data_real/romanized/tanglish.jsonl"),
}

ALGOS = ("bpe", "unigram")


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


def latin_runs(docs: Sequence[str]) -> List[str]:
    out = []
    for t in docs:
        for r in segment(t, slot_map=DEFAULT_SLOT_MAP, structural_slot=CODE_MATH,
                         whitespace_ownership="leading"):
            if r.slot == "LATIN":
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
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--train-chars", type=int, default=3_000_000)
    ap.add_argument("--held-chars", type=int, default=600_000)
    ap.add_argument("--out", default="artifacts/latin_mix_benchmark.json")
    args = ap.parse_args(argv)

    parts_train: Dict[str, List[str]] = {}
    parts_held: Dict[str, List[str]] = {}
    for name, (tr_path, he_path) in SOURCES.items():
        if not os.path.exists(tr_path):
            print(f"  {name}: missing {tr_path}, skipping")
            continue
        if tr_path == he_path:
            docs = load_docs(tr_path, args.train_chars + args.held_chars)
            cut = max(1, int(len(docs) * 0.85))
            tr, he = docs[:cut], docs[cut:]
        else:
            tr = load_docs(tr_path, args.train_chars)
            he = load_docs(he_path, args.held_chars)
        parts_train[name] = latin_runs(tr)
        parts_held[name] = latin_runs(he)
        words = sum(len(r.split()) for r in parts_held[name])
        print(f"  {name:<10} train_runs={len(parts_train[name]):>7} "
              f"({sum(len(r) for r in parts_train[name])} ch)  "
              f"held_runs={len(parts_held[name]):>6} ({words} words)")

    if not parts_train:
        print("no data")
        return 1

    pooled_train = [r for rs in parts_train.values() for r in rs]
    print(f"\n  pooled LATIN slot: {len(pooled_train)} runs "
          f"({sum(len(r) for r in pooled_train)} chars)")

    rows: List[dict] = []
    for algo in ALGOS:
        t0 = time.time()
        slot = train_slot("LATIN", algo, pooled_train, args.vocab,
                          alphabet_scope=ALPHABET_OBSERVED_ASCII,
                          max_piece_length=48)
        sec = round(time.time() - t0, 1)
        print(f"\n  {algo}: pieces={slot.local_size} ({sec}s)")
        for name, held in parts_held.items():
            words = sum(len(r.split()) for r in held)
            toks = eval_tokens(slot, held)
            rows.append({"source": name, "algo": algo, "pieces": slot.local_size,
                         "tokens": toks, "words": words,
                         "tokens_per_word": round(toks / words, 4), "sec": sec})
            print(f"     {name:<10} tok/word={toks/words:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== LATIN slot, held-out tokens/word (lower better) ===")
    print(f"{'source':<12}{'bpe':>9}{'unigram':>10}{'winner':>10}{'margin':>9}")
    by = {}
    for r in rows:
        by.setdefault(r["source"], {})[r["algo"]] = r
    for src in by:
        if len(by[src]) < 2:
            continue
        b = by[src]["bpe"]["tokens_per_word"]
        u = by[src]["unigram"]["tokens_per_word"]
        w = "unigram" if u < b else "bpe"
        print(f"{src:<12}{b:>9.4f}{u:>10.4f}{w:>10}"
              f"{abs(b-u)/max(b,u)*100:>8.1f}%")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

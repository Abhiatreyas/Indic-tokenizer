"""BPE vs Unigram across the major Indic scripts, plus romanized code-mixed Latin.

Answers two separate questions that the two-language test could not:

  1. Does "Unigram with max_piece_length=48" hold for scripts other than
     Devanagari and Kannada?
  2. Does it hold when a slot is SHARED across several languages (the design's
     actual configuration), or only when a slot holds one language?

Both tokenizers are HF `tokenizers`, trained on identical runs and scored on
identical held-out runs. SentencePiece is omitted because the provider benchmark
(diagnostics/README.md section 13) showed its BPE within 2% of HF's and its
Unigram behind HF's tuned Unigram, so it adds cost without changing the decision.

    python diagnostics/benchmark_indic_all.py --vocab 16000 --train-chars 1500000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

from tokenizer.scripts import CODE_MATH, COMMON, LATIN, NEUTRAL, segment
from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    UnrepresentableRun,
    train_slot,
)


@dataclass
class Lang:
    code: str
    name: str
    script: str
    train: str
    held: str


ML = "data_real/multilang"
RO = "data_real/romanized"

LANGS: Dict[str, Lang] = {
    "hi": Lang("hi", "Hindi", "DEVANAGARI", "data_real/hi_train.jsonl",
               "data_real/hi_heldout.jsonl"),
    "mr": Lang("mr", "Marathi", "DEVANAGARI", f"{ML}/mr.jsonl", f"{ML}/mr.jsonl"),
    "ne": Lang("ne", "Nepali", "DEVANAGARI", f"{ML}/ne.jsonl", f"{ML}/ne.jsonl"),
    "bn": Lang("bn", "Bengali", "BENGALI", f"{ML}/bn.jsonl", f"{ML}/bn.jsonl"),
    "ta": Lang("ta", "Tamil", "TAMIL", f"{ML}/ta.jsonl", f"{ML}/ta.jsonl"),
    "te": Lang("te", "Telugu", "TELUGU", f"{ML}/te.jsonl", f"{ML}/te.jsonl"),
    "ml": Lang("ml", "Malayalam", "MALAYALAM", f"{ML}/ml.jsonl", f"{ML}/ml.jsonl"),
    "gu": Lang("gu", "Gujarati", "GUJARATI", f"{ML}/gu.jsonl", f"{ML}/gu.jsonl"),
    "pa": Lang("pa", "Punjabi", "GURMUKHI", f"{ML}/pa.jsonl", f"{ML}/pa.jsonl"),
    "or": Lang("or", "Odia", "ORIYA", f"{ML}/or.jsonl", f"{ML}/or.jsonl"),
    "ur": Lang("ur", "Urdu", "ARABIC", f"{ML}/ur.jsonl", f"{ML}/ur.jsonl"),
    "kn": Lang("kn", "Kannada", "KANNADA", "data_real/kn_train.jsonl",
               "data_real/kn_heldout.jsonl"),
}

# Which languages share a script -> which share a slot, per the design.
SCRIPT_GROUPS: Dict[str, List[str]] = {}
for _c, _l in LANGS.items():
    SCRIPT_GROUPS.setdefault(_l.script, []).append(_c)

# DEFAULT_SLOT_MAP only routes COMMON/LATIN/DEVANAGARI/KANNADA; every other
# script falls through to CATCHALL, so filtering runs by slot name silently
# matched nothing for Bengali, Tamil, Telugu, Malayalam, Gujarati, Gurmukhi,
# Odia and Urdu. Building a map with one slot per script makes the extraction
# reflect the design, where adding a script means adding a slot.
SLOT_MAP: Dict[str, str] = {COMMON: NEUTRAL, LATIN: LATIN}
for _script in SCRIPT_GROUPS:
    SLOT_MAP[_script] = _script

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


def split_docs(docs: Sequence[str], hold_frac: float = 0.15):
    cut = max(1, int(len(docs) * (1 - hold_frac)))
    return list(docs[:cut]), list(docs[cut:])


def script_runs(docs: Sequence[str], script: str) -> List[str]:
    out = []
    for t in docs:
        for r in segment(t, slot_map=SLOT_MAP, structural_slot=CODE_MATH,
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


def train(runs: Sequence[str], algo: str, vocab: int):
    return train_slot(f"{algo}_slot", algo, list(runs), vocab,
                      alphabet_scope=ALPHABET_OBSERVED_ASCII,
                      max_piece_length=48)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--train-chars", type=int, default=1_500_000)
    ap.add_argument("--held-chars", type=int, default=400_000)
    ap.add_argument("--out", default="artifacts/indic_all_benchmark.json")
    ap.add_argument("--skip-script-mode", action="store_true")
    args = ap.parse_args(argv)

    rows: List[dict] = []

    # ---------------------------------------------------------------- per lang
    print("=" * 78)
    print("MODE A: one vocabulary per LANGUAGE")
    print("=" * 78)
    prepared = {}
    for code, L in LANGS.items():
        if not (os.path.exists(L.train) and os.path.exists(L.held)):
            print(f"  {code} ({L.name}): missing corpus -> {L.train}")
            continue
        if L.train == L.held:
            tr, he = split_docs(load_docs(L.train, args.train_chars + args.held_chars))
        else:
            tr = load_docs(L.train, args.train_chars)
            he = load_docs(L.held, args.held_chars)
        train_runs = script_runs(tr, L.script)
        held_runs = script_runs(he, L.script)
        if len(train_runs) < 100 or len(held_runs) < 50:
            print(f"  {code} ({L.name}): too few {L.script} runs "
                  f"({len(train_runs)}/{len(held_runs)}), skipping")
            continue
        prepared[code] = (L, train_runs, held_runs)
        words = sum(len(r.split()) for r in held_runs)
        print(f"\n  {code} ({L.name}, {L.script}): train_runs={len(train_runs)} "
              f"({sum(len(r) for r in train_runs)} ch), held_runs={len(held_runs)} "
              f"({words} words)")

        for algo in ALGOS:
            t0 = time.time()
            slot = train(train_runs, algo, args.vocab)
            toks = eval_tokens(slot, held_runs)
            rows.append({
                "mode": "per_language", "lang": code, "name": L.name,
                "script": L.script, "algo": algo, "vocab": args.vocab,
                "pieces": slot.local_size, "tokens": toks, "words": words,
                "tokens_per_word": round(toks / words, 4),
                "sec": round(time.time() - t0, 1),
            })
            print(f"     {algo:<8} pieces={slot.local_size:<6} "
                  f"tok/word={toks/words:.4f}  ({time.time()-t0:.0f}s)")

    # -------------------------------------------------------------- per script
    if not args.skip_script_mode:
        print("\n" + "=" * 78)
        print("MODE B: one vocabulary per SCRIPT, shared across its languages")
        print("=" * 78)
        for script, codes in sorted(SCRIPT_GROUPS.items()):
            present = [c for c in codes if c in prepared]
            if len(present) < 2:
                continue
            pooled = [r for c in present for r in prepared[c][1]]
            print(f"\n  {script}: pooling {present} -> {len(pooled)} runs "
                  f"({sum(len(r) for r in pooled)} chars)")
            for algo in ALGOS:
                t0 = time.time()
                slot = train(pooled, algo, args.vocab)
                print(f"     {algo:<8} pieces={slot.local_size:<6} "
                      f"({time.time()-t0:.0f}s)")
                for c in present:
                    L, _, held = prepared[c]
                    words = sum(len(r.split()) for r in held)
                    toks = eval_tokens(slot, held)
                    rows.append({
                        "mode": "per_script", "lang": c, "name": L.name,
                        "script": script, "algo": algo, "vocab": args.vocab,
                        "pieces": slot.local_size, "tokens": toks, "words": words,
                        "tokens_per_word": round(toks / words, 4),
                        "sec": round(time.time() - t0, 1),
                    })
                    print(f"        {c:<4} {L.name:<10} tok/word={toks/words:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"RESULT: held-out tokens/word at vocab {args.vocab} (lower better)")
    print("=" * 78)
    print(f"{'mode':<13}{'lang':<6}{'script':<13}{'bpe':>9}{'unigram':>10}"
          f"{'winner':>10}{'margin':>9}")
    by_key: Dict[tuple, dict] = {}
    for r in rows:
        by_key.setdefault((r["mode"], r["lang"]), {})[r["algo"]] = r
    wins = {"bpe": 0, "unigram": 0, "tie": 0}
    for key in sorted(by_key):
        d = by_key[key]
        if "bpe" not in d or "unigram" not in d:
            continue
        b, u = d["bpe"]["tokens_per_word"], d["unigram"]["tokens_per_word"]
        w = "unigram" if u < b else "bpe"
        wins[w] += 1
        margin = abs(b - u) / max(b, u) * 100
        print(f"{key[0]:<13}{key[1]:<6}{d['bpe']['script']:<13}"
              f"{b:>9.4f}{u:>10.4f}{w:>10}{margin:>8.1f}%")
    print(f"\n  unigram wins {wins['unigram']}, bpe wins {wins['bpe']}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Re-test Unigram vs BPE for Indic slots, on real text.

The original decision (SPEC.md section 5) was made on a synthetic corpus holding
~101 distinct Hindi word forms and 75 distinct Devanagari runs. Unigram's EM needs
lexical diversity, so that was close to a worst case for it, and the follow-up
diagnostic that "repeated the corpus 20x" added occurrences rather than words.

This re-runs the comparison on real corpora -- Kannada from the local CulturaX /
held-out sets, Hindi from Wikipedia -- across vocabulary budgets and both
whitespace modes, measuring held-out tokens per character AND tokens per word
using the same per-character fallback the partitioner uses.

    python diagnostics/compare_indic_algorithms.py --train-chars 15000000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Sequence, Tuple

from tokenizer.scripts import CODE_MATH, DEFAULT_SLOT_MAP, segment
from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    ByteFallbackSlot,
    UnrepresentableRun,
    train_slot,
)

SCRIPTS = {"hi": "DEVANAGARI", "kn": "KANNADA"}
TRAIN_FILES = {"hi": "data_real/hi_train.jsonl", "kn": "data_real/kn_train.jsonl"}
HELD_FILES = {"hi": "data_real/hi_heldout.jsonl", "kn": "data_real/kn_heldout.jsonl"}


def load_texts(path: str, limit_chars: int | None = None) -> List[str]:
    out: List[str] = []
    total = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)["text"]
            out.append(t)
            total += len(t)
            if limit_chars is not None and total >= limit_chars:
                break
    return out


def script_runs(texts: Sequence[str], script: str, ownership: str) -> List[str]:
    """The runs this script's slot would actually be trained on."""
    out: List[str] = []
    for t in texts:
        for r in segment(
            t,
            slot_map=DEFAULT_SLOT_MAP,
            structural_slot=CODE_MATH,
            whitespace_ownership=ownership,
        ):
            if r.slot == script:
                out.append(r.text)
    return out


def count_tokens(slot, catchall, runs: Sequence[str]) -> Tuple[int, int]:
    """Tokens for `runs`, using the partitioner's per-character fallback.

    Mirrors PartitionedTokenizer._encode_text so the measurement is what the
    pipeline actually does, not an idealisation.
    """
    total = 0
    fallback_chars = 0
    for text in runs:
        stack = [text]
        while stack:
            chunk = stack.pop()
            if not chunk:
                continue
            try:
                total += len(slot.encode(chunk))
                continue
            except UnrepresentableRun:
                pass
            if len(chunk) == 1:
                fallback_chars += 1
                total += len(catchall.encode(chunk))
                continue
            mid = len(chunk) // 2
            stack.append(chunk[mid:])
            stack.append(chunk[:mid])
    return total, fallback_chars


def word_count(runs: Sequence[str]) -> int:
    return sum(len(r.split()) for r in runs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-chars", type=int, default=15_000_000)
    ap.add_argument("--heldout-chars", type=int, default=4_000_000)
    ap.add_argument("--vocabs", default="2000,8000,32000")
    ap.add_argument("--langs", default="hi,kn")
    ap.add_argument("--out", default="artifacts/indic_recheck.json")
    args = ap.parse_args(argv)

    vocabs = [int(v) for v in args.vocabs.split(",")]
    catchall = ByteFallbackSlot()
    results: List[dict] = []

    for lang in [s for s in args.langs.split(",") if s.strip()]:
        script = SCRIPTS[lang]
        train_path, held_path = TRAIN_FILES[lang], HELD_FILES[lang]
        if not (os.path.exists(train_path) and os.path.exists(held_path)):
            print(f"  {lang}: missing {train_path} or {held_path}, skipping")
            continue

        train_texts = load_texts(train_path, args.train_chars)
        held_texts = load_texts(held_path, args.heldout_chars)
        print(f"\n=== {lang} ({script}) ===")
        print(
            f"  real corpus: train={len(train_texts)} docs "
            f"({sum(len(t) for t in train_texts)} chars), "
            f"heldout={len(held_texts)} docs "
            f"({sum(len(t) for t in held_texts)} chars)"
        )

        for ownership in ("neutral", "leading"):
            train_runs = script_runs(train_texts, script, ownership)
            held_runs = script_runs(held_texts, script, ownership)
            held_chars = sum(len(r) for r in held_runs)
            held_words = word_count(held_runs)
            print(
                f"  [{ownership}] train_runs={len(train_runs)} "
                f"distinct_runs={len(set(train_runs))} "
                f"distinct_words={len({w for r in train_runs for w in r.split()})} "
                f"| held_runs={len(held_runs)} chars={held_chars} words={held_words}"
            )

            for vocab in vocabs:
                row = {
                    "lang": lang, "script": script, "ownership": ownership,
                    "vocab": vocab, "held_chars": held_chars,
                    "held_words": held_words, "train_runs": len(train_runs),
                    "distinct_runs": len(set(train_runs)),
                    "distinct_words": len({w for r in train_runs for w in r.split()}),
                }
                for algo in ("bpe", "unigram"):
                    t0 = time.time()
                    try:
                        slot = train_slot(
                            f"{script}_{algo}", algo, train_runs, vocab,
                            alphabet_scope=ALPHABET_OBSERVED_ASCII,
                        )
                        toks, fb = count_tokens(slot, catchall, held_runs)
                        row[algo] = {
                            "actual_vocab": slot.local_size,
                            "tokens": toks,
                            "tokens_per_char": round(toks / held_chars, 4),
                            "tokens_per_word": round(toks / held_words, 3),
                            "fallback_chars": fb,
                            "train_sec": round(time.time() - t0, 1),
                        }
                    except Exception as e:  # noqa: BLE001 - report, do not crash
                        row[algo] = {"error": f"{type(e).__name__}: {e}"}

                b, u = row.get("bpe", {}), row.get("unigram", {})
                if "tokens_per_char" in b and "tokens_per_char" in u:
                    row["winner"] = (
                        "unigram"
                        if u["tokens_per_char"] < b["tokens_per_char"]
                        else "bpe"
                    )
                    row["unigram_gain_pct"] = round(
                        100 * (b["tokens_per_char"] - u["tokens_per_char"])
                        / b["tokens_per_char"], 2,
                    )
                results.append(row)
                print(
                    f"    vocab={vocab:<6} "
                    f"bpe={b.get('tokens_per_word')} tok/word "
                    f"({b.get('actual_vocab')}, {b.get('train_sec')}s)   "
                    f"unigram={u.get('tokens_per_word')} tok/word "
                    f"({u.get('actual_vocab')}, {u.get('train_sec')}s)   "
                    f"-> {row.get('winner')} "
                    f"({row.get('unigram_gain_pct')}% unigram gain)"
                )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== held-out tokens/word on REAL text (lower is better) ===")
    print(f"{'lang':<5}{'ws':<9}{'vocab':<8}{'bpe':>8}{'unigram':>9}{'winner':>10}{'gain%':>8}")
    for r in results:
        b, u = r.get("bpe", {}), r.get("unigram", {})
        print(
            f"{r['lang']:<5}{r['ownership']:<9}{r['vocab']:<8}"
            f"{b.get('tokens_per_word', float('nan')):>8}"
            f"{u.get('tokens_per_word', float('nan')):>9}"
            f"{r.get('winner', '?'):>10}{r.get('unigram_gain_pct', 0):>8}"
        )
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

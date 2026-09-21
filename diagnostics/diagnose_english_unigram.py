"""Why does English score better under Unigram? Isolating the confounds.

In section 14 the LATIN slot held English + Hinglish + Tanglish POOLED, and
English improved 8.0% under Unigram. Three candidate explanations:

  H1 algorithm   -- Unigram is genuinely better for English, independent of
                    pooling. Plausible: Unigram optimises corpus likelihood
                    globally, where BPE is a greedy frequency merge, and
                    SentencePiece Unigram is what T5 and ALBERT used for English.
  H2 pooling     -- the pooled slot simply gives Unigram more training data
                    (5.2M chars vs 2.85M English-only) and Unigram benefits more.
  H3 piece limit -- Unigram only wins because max_piece_length was raised to 48;
                    at the default 16 it loses. Then the finding is about the
                    setting, not the algorithm.

Conditions below separate them: English ALONE at both the default and raised
piece limit, at two vocabularies, plus the pooled slot at the same vocabularies.

    python diagnostics/diagnose_english_unigram.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    UnrepresentableRun,
    train_slot,
)

SOURCES = {
    "english": ("data_real/en_train.jsonl", "data_real/en_heldout.jsonl"),
    "hinglish": ("data_real/romanized/hinglish.jsonl", "data_real/romanized/hinglish.jsonl"),
    "tanglish": ("data_real/romanized/tanglish.jsonl", "data_real/romanized/tanglish.jsonl"),
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


def latin_runs(docs: Sequence[str]) -> List[str]:
    from tokenizer.scripts import CODE_MATH, DEFAULT_SLOT_MAP, segment

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


def run(train_runs, eval_runs, words, algo, vocab, mpl, label, rows):
    t0 = time.time()
    slot = train_slot("LATIN", algo, list(train_runs), vocab,
                      alphabet_scope=ALPHABET_OBSERVED_ASCII,
                      max_piece_length=mpl)
    toks = eval_tokens(slot, eval_runs)
    rec = {"label": label, "algo": algo, "vocab": vocab, "mpl": mpl,
           "train_runs": len(train_runs),
           "train_chars": sum(len(r) for r in train_runs),
           "pieces": slot.local_size, "tokens": toks, "words": words,
           "tokens_per_word": round(toks / words, 4),
           "sec": round(time.time() - t0, 1)}
    rows.append(rec)
    print(f"  {label:<34} {algo:<8} mpl={mpl:<3} pieces={slot.local_size:<6} "
          f"tok/word={toks/words:.4f}  ({time.time()-t0:.0f}s)")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-chars", type=int, default=2_800_000)
    ap.add_argument("--held-chars", type=int, default=600_000)
    ap.add_argument("--vocabs", default="16000,32000")
    ap.add_argument("--out", default="artifacts/english_unigram_diag.json")
    args = ap.parse_args(argv)

    en_tr = latin_runs(load_docs(SOURCES["english"][0], args.train_chars))
    en_he = latin_runs(load_docs(SOURCES["english"][1], args.held_chars))
    en_words = sum(len(r.split()) for r in en_he)
    print(f"English-only: train_runs={len(en_tr)} "
          f"({sum(len(r) for r in en_tr)} chars), held_runs={len(en_he)} "
          f"({en_words} words)\n")

    pooled = list(en_tr)
    for name in ("hinglish", "tanglish"):
        pooled += latin_runs(load_docs(SOURCES[name][0], args.train_chars))
    print(f"Pooled LATIN: {len(pooled)} runs "
          f"({sum(len(r) for r in pooled)} chars)\n")

    rows: List[Dict] = []
    for vocab in [int(v) for v in args.vocabs.split(",")]:
        print(f"--- vocab {vocab} ---")
        print(" [A] English ALONE (isolates H1 and H3)")
        run(en_tr, en_he, en_words, "bpe", vocab, 48,
            f"english_only bpe", rows)
        run(en_tr, en_he, en_words, "unigram", vocab, 16,
            f"english_only unigram mpl16", rows)
        run(en_tr, en_he, en_words, "unigram", vocab, 48,
            f"english_only unigram mpl48", rows)

        print(" [B] POOLED slot, evaluated on English only (isolates H2)")
        run(pooled, en_he, en_words, "bpe", vocab, 48,
            f"pooled bpe", rows)
        run(pooled, en_he, en_words, "unigram", vocab, 48,
            f"pooled unigram mpl48", rows)
        print()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("=== summary: English held-out tokens/word (lower better) ===")
    print(f"{'vocab':<8}{'condition':<34}{'algo':<10}{'mpl':>4}{'tok/word':>10}")
    for r in sorted(rows, key=lambda x: (x["vocab"], x["tokens_per_word"])):
        print(f"{r['vocab']:<8}{r['label']:<34}{r['algo']:<10}{r['mpl']:>4}"
              f"{r['tokens_per_word']:>10.4f}")

    print("\n=== interpretation ===")
    for vocab in sorted({r["vocab"] for r in rows}):
        d = {r["label"]: r for r in rows if r["vocab"] == vocab}
        e_bpe = d["english_only bpe"]["tokens_per_word"]
        e_u16 = d["english_only unigram mpl16"]["tokens_per_word"]
        e_u48 = d["english_only unigram mpl48"]["tokens_per_word"]
        p_bpe = d["pooled bpe"]["tokens_per_word"]
        p_u48 = d["pooled unigram mpl48"]["tokens_per_word"]
        print(f"vocab {vocab}:")
        print(f"  H3 piece limit : english-only unigram mpl16={e_u16:.4f} vs "
              f"mpl48={e_u48:.4f}  -> "
              f"{'limit is what matters' if e_u16 > e_u48 * 1.02 else 'limit is NOT the main effect'}")
        print(f"  H1 algorithm   : english-only unigram(mpl48)={e_u48:.4f} vs "
              f"bpe={e_bpe:.4f}  -> "
              f"{'unigram wins alone' if e_u48 < e_bpe else 'unigram does NOT win alone'}"
              f" ({(e_bpe-e_u48)/e_bpe*100:+.1f}%)")
        print(f"  H2 pooling     : pooled unigram={p_u48:.4f} vs "
              f"english-only unigram={e_u48:.4f}  -> "
              f"{(e_u48-p_u48)/e_u48*100:+.1f}% from pooling")
        print(f"     and pooled bpe={p_bpe:.4f} vs english-only bpe={e_bpe:.4f}"
              f"  -> {(e_bpe-p_bpe)/e_bpe*100:+.1f}% from pooling")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

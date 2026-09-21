"""Evaluate the Vachana Kannada tokenizers against my HF BPE/Unigram slots.

Motor question: Vachana's own tokenizer reportedly beat BPE comfortably on
Kannada. If that is reproducible on the same held-out text, then my conclusion
(and/or the HF tokenizers implementation I used) is wrong.

All tokenizers are scored on the SAME real Kannada held-out text, using tokens per
whitespace-delimited word.

    python diagnostics/evaluate_vachana_tokenizer.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Sequence

from tokenizer.scripts import (
    CODE_MATH,
    DEFAULT_SLOT_MAP,
    KANNADA,
    segment,
)
from tokenizer.train import ALPHABET_OBSERVED_ASCII, train_slot

VACHANA_DIR = r"E:\VachanaLLM\tokenizer\models"
VACHANA_MODELS = {
    "vachana_unigram": os.path.join(VACHANA_DIR, "kannada_32k_unigram.model"),
    "vachana_unigram_v2": os.path.join(VACHANA_DIR, "kannada_32k_unigram_v2.model"),
    "vachana_bpe": os.path.join(VACHANA_DIR, "kannada_32k_bpe.model"),
    "sarvam1": os.path.join(VACHANA_DIR, "sarvam1", "tokenizer.model"),
}


def load_docs(path: str, char_cap: int) -> List[str]:
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
            if total >= char_cap:
                break
    return out


def score(encode: Callable[[str], Sequence], docs: Sequence[str]) -> Dict:
    tokens = 0
    words = 0
    chars = 0
    for d in docs:
        tokens += len(encode(d))
        words += len(d.split())
        chars += len(d)
    return {
        "tokens": tokens,
        "words": words,
        "chars": chars,
        "tokens_per_word": round(tokens / words, 3),
        "tokens_per_char": round(tokens / chars, 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout", default="data_real/kn_heldout.jsonl")
    ap.add_argument("--heldout-chars", type=int, default=2_000_000)
    ap.add_argument("--train", default="data_real/kn_train.jsonl")
    ap.add_argument("--train-chars", type=int, default=5_000_000)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--out", default="artifacts/vachana_comparison.json")
    args = ap.parse_args(argv)

    held = load_docs(args.heldout, args.heldout_chars)
    print(f"held-out: {len(held)} docs, {sum(len(d) for d in held)} chars")
    print()

    rows: List[dict] = []

    # ---- Vachana's own tokenizers (SentencePiece) ------------------------
    import sentencepiece as spm

    for name, path in VACHANA_MODELS.items():
        if not os.path.exists(path):
            print(f"  {name}: missing {path}")
            continue
        sp = spm.SentencePieceProcessor(model_file=path)
        r = score(lambda t, sp=sp: sp.encode(t, out_type=int), held)
        r.update({"tokenizer": name, "kind": "sentencepiece", "pieces": sp.get_piece_size()})
        rows.append(r)
        print(
            f"  {name:<20} pieces={sp.get_piece_size():<7} "
            f"tokens/word={r['tokens_per_word']:<7} tokens/char={r['tokens_per_char']}"
        )

    # ---- my HF slots, trained on the same held-out's language ------------
    train_docs = load_docs(args.train, args.train_chars)
    runs: List[str] = []
    for t in train_docs:
        for run in segment(
            t,
            slot_map=DEFAULT_SLOT_MAP,
            structural_slot=CODE_MATH,
            whitespace_ownership="leading",
        ):
            if run.slot == KANNADA:
                runs.append(run.text)
    print(f"\n  trained on {len(runs)} Kannada runs "
          f"({sum(len(r) for r in runs)} chars), leading ownership")

    for algo in ("bpe", "unigram"):
        t0 = time.time()
        slot = train_slot(
            f"KANNADA_{algo}", algo, runs, args.vocab,
            alphabet_scope=ALPHABET_OBSERVED_ASCII,
        )
        # HF slots emit no whitespace markers of their own; encode whole docs so
        # spaces are represented the way the pipeline would (via byte chars).
        r = score(lambda t, s=slot: s.encode(t), held)
        r.update({
            "tokenizer": f"hf_{algo}",
            "kind": "hf_tokenizers",
            "pieces": slot.local_size,
            "train_sec": round(time.time() - t0, 1),
        })
        rows.append(r)
        print(
            f"  {'hf_' + algo:<20} pieces={slot.local_size:<7} "
            f"tokens/word={r['tokens_per_word']:<7} tokens/char={r['tokens_per_char']}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== tokens per word on identical real Kannada held-out (lower better) ===")
    for r in sorted(rows, key=lambda x: x["tokens_per_word"]):
        print(
            f"  {r['tokenizer']:<20} {r['kind']:<16} pieces={r['pieces']:<7} "
            f"tokens/word={r['tokens_per_word']}"
        )
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

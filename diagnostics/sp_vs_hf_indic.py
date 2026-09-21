"""SentencePiece vs HF `tokenizers` for the Indic algorithm choice.

Two competing explanations for why Vachana measured Unigram > BPE for Kannada
(1.7295 vs 1.7536, a 1.37% edge) while my pipeline measured BPE > Unigram:

  H1  the algorithm really is a wash, and my result is an artefact of HF's
      UnigramTrainer being weaker than SentencePiece's reference implementation;
  H2  something about corpora / vocabulary sharing / measurement differs.

This tests H1 directly: train SentencePiece Unigram AND SentencePiece BPE on my
exact Kannada training runs with Vachana's reported settings, then score all
tokenizers on the identical held-out run strings.

    python diagnostics/sp_vs_hf_indic.py --train-chars 5000000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Sequence

import sentencepiece as spm

from tokenizer.scripts import CODE_MATH, DEFAULT_SLOT_MAP, KANNADA, segment
from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    ByteFallbackSlot,
    UnrepresentableRun,
    train_slot,
)

VACHANA_DIR = r"E:\VachanaLLM\tokenizer\models"
PRETRAINED = {
    "vachana_sp_unigram": os.path.join(VACHANA_DIR, "kannada_32k_unigram.model"),
    "vachana_sp_bpe": os.path.join(VACHANA_DIR, "kannada_32k_bpe.model"),
    "sarvam1_shared_10lang": os.path.join(VACHANA_DIR, "sarvam1", "tokenizer.model"),
}

# Settings copied from Vachana's TOKENIZER_REPORT.md so the comparison is of
# implementations, not of configuration.
SP_SETTINGS = dict(
    vocab_size=32000,
    character_coverage=0.9999,
    byte_fallback=True,
    split_digits=True,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    max_sentence_length=8192,
    user_defined_symbols=["<pad>"],
)


def load_texts(path: str, cap: int) -> List[str]:
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


def kannada_runs(texts: Sequence[str], ownership: str = "leading") -> List[str]:
    runs = []
    for t in texts:
        for r in segment(
            t,
            slot_map=DEFAULT_SLOT_MAP,
            structural_slot=CODE_MATH,
            whitespace_ownership=ownership,
        ):
            if r.slot == KANNADA:
                runs.append(r.text)
    return runs


def count_hf(slot, catchall, runs: Sequence[str]):
    """Tokens for HF slots with the partitioner's per-character fallback."""
    total = 0
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
                total += len(catchall.encode(chunk))
                continue
            mid = len(chunk) // 2
            stack.append(chunk[mid:])
            stack.append(chunk[:mid])
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data_real/kn_train.jsonl")
    ap.add_argument("--heldout", default="data_real/kn_heldout.jsonl")
    ap.add_argument("--train-chars", type=int, default=5_000_000)
    ap.add_argument("--heldout-chars", type=int, default=2_000_000)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--workdir", default="artifacts/sp_vs_hf")
    ap.add_argument("--out", default="artifacts/sp_vs_hf_results.json")
    args = ap.parse_args(argv)

    os.makedirs(args.workdir, exist_ok=True)
    train_runs = kannada_runs(load_texts(args.train, args.train_chars))
    held_runs = kannada_runs(load_texts(args.heldout, args.heldout_chars))
    train_chars = sum(len(r) for r in train_runs)
    held_chars = sum(len(r) for r in held_runs)
    held_words = sum(len(r.split()) for r in held_runs)
    print(
        f"Kannada runs: train={len(train_runs)} ({train_chars} chars), "
        f"heldout={len(held_runs)} ({held_chars} chars, {held_words} words)"
    )

    train_txt = os.path.join(args.workdir, "sp_train.txt")
    with open(train_txt, "w", encoding="utf-8") as fh:
        for r in train_runs:
            fh.write(r.replace("\n", " ") + "\n")

    rows: List[Dict] = []

    def record(name: str, kind: str, pieces: int, count: Callable[[], int]):
        toks = count()
        rows.append({
            "tokenizer": name, "kind": kind, "pieces": pieces,
            "tokens": toks,
            "tokens_per_char": round(toks / held_chars, 4),
            "tokens_per_word": round(toks / held_words, 4),
        })
        print(f"  {name:<24} pieces={pieces:<7} tokens/word={toks / held_words:.4f}")

    # ---- SentencePiece, trained by me with Vachana's settings ------------
    print("\n[SentencePiece, trained here on the same runs]")
    for algo in ("unigram", "bpe"):
        t0 = time.time()
        prefix = os.path.join(args.workdir, f"sp_{algo}")
        spm.SentencePieceTrainer.train(
            input=train_txt, model_prefix=prefix, model_type=algo,
            **{**SP_SETTINGS, "vocab_size": args.vocab},
        )
        sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
        record(f"sp_{algo}_mine", "sentencepiece", sp.get_piece_size(),
               lambda sp=sp: sum(len(sp.encode(r, out_type=int)) for r in held_runs))
        print(f"      trained in {time.time() - t0:.1f}s")

    # ---- HF tokenizers, same runs ---------------------------------------
    print("\n[HF tokenizers, trained here on the same runs]")
    catchall = ByteFallbackSlot()
    for algo in ("bpe", "unigram"):
        t0 = time.time()
        slot = train_slot(f"KANNADA_{algo}", algo, train_runs, args.vocab,
                          alphabet_scope=ALPHABET_OBSERVED_ASCII)
        record(f"hf_{algo}_mine", "hf_tokenizers", slot.local_size,
               lambda s=slot: count_hf(s, catchall, held_runs))
        print(f"      trained in {time.time() - t0:.1f}s")

    # ---- pretrained references ------------------------------------------
    print("\n[pretrained references]")
    for name, path in PRETRAINED.items():
        if not os.path.exists(path):
            print(f"  {name}: missing {path}")
            continue
        sp = spm.SentencePieceProcessor(model_file=path)
        record(name, "sentencepiece", sp.get_piece_size(),
               lambda sp=sp: sum(len(sp.encode(r, out_type=int)) for r in held_runs))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== tokens per word, identical Kannada held-out runs (lower better) ===")
    for r in sorted(rows, key=lambda x: x["tokens_per_word"]):
        print(f"  {r['tokenizer']:<24} {r['kind']:<16} pieces={r['pieces']:<7} "
              f"tokens/word={r['tokens_per_word']}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

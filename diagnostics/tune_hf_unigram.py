"""Is HF's Unigram deficit an implementation gap or a configuration gap?

Measured earlier: on identical Kannada runs and 32k vocab, HF's UnigramTrainer
scored 1.9210 tokens/word against SentencePiece's 1.6577 -- a 15.9% deficit.

Suspected cause: `max_piece_length`. SentencePiece's `max_sentencepiece_length`
defaults to 16 *Unicode characters*, so it can learn a 16-akshara Kannada word.
HF's `UnigramTrainer.max_piece_length` defaults to 16 *pre-tokenized tokens*, and
under a ByteLevel pre-tokenizer one Kannada akshara is ~3 tokens -- so HF is
capped at roughly 5 Kannada characters. A 32k Kannada vocabulary full of long
whole-word pieces would be unreachable.

If that is the cause, raising the limit should close most of the gap.

    python diagnostics/tune_hf_unigram.py --lang kn --vocab 32000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tokenizer.scripts import CODE_MATH, DEFAULT_SLOT_MAP, KANNADA, segment
from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    BYTE_ALPHABET,
    ByteFallbackSlot,
    UnrepresentableRun,
    observed_alphabet,
)

BYTE_LEVEL = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)


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


def runs_for(texts: Sequence[str], slot: str = KANNADA) -> List[str]:
    out = []
    for t in texts:
        for r in segment(t, slot_map=DEFAULT_SLOT_MAP, structural_slot=CODE_MATH,
                         whitespace_ownership="leading"):
            if r.slot == slot:
                out.append(r.text)
    return out


class Slot:
    def __init__(self, tok):
        self.t = tok
        vocab = tok.get_vocab()
        specials = set(tok.get_added_tokens_decoder().keys())
        ordered = sorted(vocab.items(), key=lambda kv: kv[1])
        self.content = [t for t, i in ordered if i not in specials]
        self.h2l = {}
        self.l2h = []
        for t, i in ordered:
            if i in specials:
                continue
            self.h2l[i] = len(self.l2h)
            self.l2h.append(i)
        self.local_size = len(self.content)

    def encode(self, s):
        out = []
        for hf in self.t.encode(s).ids:
            loc = self.h2l.get(hf)
            if loc is None:
                raise UnrepresentableRun(f"hf id {hf}")
            out.append(loc)
        return out


def count(slot, catchall, runs):
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
                total += len(catchall.encode(c))
                continue
            mid = len(c) // 2
            stack.append(c[mid:])
            stack.append(c[:mid])
    return total


def train_hf_unigram(runs, vocab, max_piece_length, num_sub_iterations,
                     alphabet_scope):
    alphabet = (BYTE_ALPHABET if alphabet_scope == "full"
                else observed_alphabet(runs, include_ascii=True))
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = BYTE_LEVEL
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        list(runs),
        trainer=trainers.UnigramTrainer(
            vocab_size=vocab,
            initial_alphabet=alphabet,
            special_tokens=["<unk>"],
            unk_token="<unk>",
            max_piece_length=max_piece_length,
            num_sub_iterations=num_sub_iterations,
            show_progress=False,
        ),
    )
    return Slot(tok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data_real/kn_train.jsonl")
    ap.add_argument("--heldout", default="data_real/kn_heldout.jsonl")
    ap.add_argument("--train-chars", type=int, default=5_000_000)
    ap.add_argument("--heldout-chars", type=int, default=2_000_000)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--out", default="artifacts/hf_unigram_tuning.json")
    args = ap.parse_args(argv)

    train_runs = runs_for(load_texts(args.train, args.train_chars))
    held_runs = runs_for(load_texts(args.heldout, args.heldout_chars))
    held_words = sum(len(r.split()) for r in held_runs)
    print(f"train runs={len(train_runs)}  heldout runs={len(held_runs)} "
          f"words={held_words}")

    # how long is a Kannada word in byte-level tokens?
    import statistics
    lens = sorted(len(r.encode("utf-8")) for r in train_runs)
    print(f"byte-length of training runs: median={statistics.median(lens)}, "
          f"p90={lens[int(len(lens)*0.9)]}, p99={lens[int(len(lens)*0.99)]}, "
          f"max={lens[-1]}")
    print("  (HF max_piece_length counts THESE; SentencePiece's limit of 16 "
          "counts Unicode characters)")
    print()

    catchall = ByteFallbackSlot()
    rows = []
    for mpl in (16, 32, 48, 64, 128):
        t0 = time.time()
        slot = train_hf_unigram(train_runs, args.vocab, mpl, 2, "observed+ascii")
        toks = count(slot, catchall, held_runs)
        rows.append({"max_piece_length": mpl, "num_sub_iterations": 2,
                     "pieces": slot.local_size,
                     "tokens_per_word": round(toks / held_words, 4),
                     "sec": round(time.time() - t0, 1)})
        print(f"  max_piece_length={mpl:<4} pieces={slot.local_size:<7} "
              f"tokens/word={toks/held_words:.4f}  ({time.time()-t0:.0f}s)")

    # does more EM help?
    for nsi in (4, 8):
        t0 = time.time()
        slot = train_hf_unigram(train_runs, args.vocab, 64, nsi, "observed+ascii")
        toks = count(slot, catchall, held_runs)
        rows.append({"max_piece_length": 64, "num_sub_iterations": nsi,
                     "pieces": slot.local_size,
                     "tokens_per_word": round(toks / held_words, 4),
                     "sec": round(time.time() - t0, 1)})
        print(f"  max_piece_length=64   num_sub_iterations={nsi} "
              f"pieces={slot.local_size:<7} tokens/word={toks/held_words:.4f}"
              f"  ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwritten: {args.out}")
    print("reference: SentencePiece unigram 1.6577, SentencePiece bpe 1.6181, "
          "HF bpe 1.6470, HF unigram default(16) see above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

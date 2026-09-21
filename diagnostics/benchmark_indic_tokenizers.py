"""Benchmark the maintained subword-trainer implementations on Indic text.

Provider landscape (PyPI, checked at run time):
  * `tokenizers` (HF, Rust)      -- maintained, trains BPE and Unigram
  * `sentencepiece` (Google, C++)-- maintained, trains BPE, Unigram, word, char
  * `tiktoken`                   -- inference only, NO trainer, so unusable
  * `youtokentome`, `subword-nmt`, `fastbpe` -- abandoned 2019-2021
  * SuperBPE, minbpe             -- research repos, not packaged

So the real choice is HF `tokenizers` vs SentencePiece, and the interesting
variable turns out to be configuration rather than implementation.

Every candidate is trained on identical runs and scored on identical held-out
runs of the same language. Losslessness is verified for each.

    python diagnostics/benchmark_indic_tokenizers.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Sequence

import sentencepiece as spm
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tokenizer.scripts import CODE_MATH, DEFAULT_SLOT_MAP, segment
from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    ByteFallbackSlot,
    UnrepresentableRun,
    observed_alphabet,
)

BYTE_LEVEL = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)

LANGS = {
    "kn": ("KANNADA", "data_real/kn_train.jsonl", "data_real/kn_heldout.jsonl"),
    "hi": ("DEVANAGARI", "data_real/hi_train.jsonl", "data_real/hi_heldout.jsonl"),
}

# Settings mirrored from Vachana's report so SentencePiece is at its best.
SP_SETTINGS = dict(
    character_coverage=0.9999,
    byte_fallback=True,
    split_digits=True,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    max_sentence_length=8192,
    user_defined_symbols=["<pad>"],
    minloglevel=2,
)

# HF `max_piece_length` counts PRE-TOKENIZED tokens; under ByteLevel one Indic
# akshara is ~3 of them, so the default of 16 caps pieces at ~5 aksharas.
HF_MPL_CANDIDATES = (16, 48)


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


def runs_for(texts: Sequence[str], slot: str) -> List[str]:
    out = []
    for t in texts:
        for r in segment(t, slot_map=DEFAULT_SLOT_MAP, structural_slot=CODE_MATH,
                         whitespace_ownership="leading"):
            if r.slot == slot:
                out.append(r.text)
    return out


def count_hf(slot, catchall, runs) -> int:
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


class HfSlot:
    def __init__(self, tok):
        self.t = tok
        specials = set(tok.get_added_tokens_decoder().keys())
        ordered = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
        self.content = [t for t, i in ordered if i not in specials]
        self.h2l, self.l2h = {}, []
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


def train_hf(runs, vocab, algo, mpl=16):
    alphabet = observed_alphabet(runs, include_ascii=True)
    if algo == "unigram":
        model, kwargs = models.Unigram(), {"max_piece_length": mpl}
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab, initial_alphabet=alphabet,
            special_tokens=["<unk>"], unk_token="<unk>",
            show_progress=False, **kwargs)
    else:
        model = models.BPE(unk_token="<unk>")
        trainer = trainers.BpeTrainer(
            vocab_size=vocab, initial_alphabet=alphabet,
            special_tokens=["<unk>"], show_progress=False)
    tok = Tokenizer(model)
    tok.pre_tokenizer = BYTE_LEVEL
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(list(runs), trainer=trainer)
    return HfSlot(tok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-chars", type=int, default=5_000_000)
    ap.add_argument("--heldout-chars", type=int, default=2_000_000)
    ap.add_argument("--vocabs", default="8000,32000")
    ap.add_argument("--langs", default="kn,hi")
    ap.add_argument("--out", default="artifacts/indic_benchmark.json")
    ap.add_argument("--workdir", default="artifacts/bench_sp")
    args = ap.parse_args(argv)

    vocabs = [int(v) for v in args.vocabs.split(",")]
    catchall = ByteFallbackSlot()
    os.makedirs(args.workdir, exist_ok=True)
    rows: List[Dict] = []

    for lang in [s for s in args.langs.split(",") if s.strip()]:
        slot_name, train_path, held_path = LANGS[lang]
        if not (os.path.exists(train_path) and os.path.exists(held_path)):
            print(f"{lang}: missing corpus, skipping")
            continue
        train_runs = runs_for(load_texts(train_path, args.train_chars), slot_name)
        held_runs = runs_for(load_texts(held_path, args.heldout_chars), slot_name)
        held_words = sum(len(r.split()) for r in held_runs)
        held_chars = sum(len(r) for r in held_runs)

        train_txt = os.path.join(args.workdir, f"{lang}_sp_train.txt")
        with open(train_txt, "w", encoding="utf-8") as fh:
            for r in train_runs:
                fh.write(r.replace("\n", " ") + "\n")

        print(f"\n=== {lang} ({slot_name}) ===")
        print(f"  train runs={len(train_runs)}  heldout runs={len(held_runs)} "
              f"words={held_words} chars={held_chars}")

        for vocab in vocabs:
            print(f"\n  -- vocab {vocab} --")

            def record(name, kind, pieces, tokens, sec, rt_ok):
                rows.append({
                    "lang": lang, "vocab": vocab, "tokenizer": name, "kind": kind,
                    "pieces": pieces, "tokens": tokens,
                    "tokens_per_word": round(tokens / held_words, 4),
                    "tokens_per_char": round(tokens / held_chars, 4),
                    "sec": round(sec, 1), "lossless": rt_ok,
                })
                print(f"     {name:<26} pieces={pieces:<7} "
                      f"tokens/word={tokens/held_words:.4f} "
                      f"lossless={rt_ok} ({sec:.0f}s)")

            # ---- SentencePiece unigram + bpe ------------------------------
            for algo in ("unigram", "bpe"):
                t0 = time.time()
                prefix = os.path.join(args.workdir, f"{lang}_{algo}_{vocab}")
                spm.SentencePieceTrainer.train(
                    input=train_txt, model_prefix=prefix, model_type=algo,
                    vocab_size=vocab, **SP_SETTINGS)
                sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
                toks = sum(len(sp.encode(r, out_type=int)) for r in held_runs)
                ok = all(sp.decode(sp.encode(r, out_type=int)) == r
                         for r in held_runs[:3000])
                record(f"sp_{algo}", "sentencepiece", sp.get_piece_size(),
                       toks, time.time() - t0, ok)

            # ---- HF unigram (default and tuned) + bpe ---------------------
            for mpl in HF_MPL_CANDIDATES:
                t0 = time.time()
                slot = train_hf(train_runs, vocab, "unigram", mpl)
                toks = count_hf(slot, catchall, held_runs)
                label = "hf_unigram_default" if mpl == 16 else f"hf_unigram_mpl{mpl}"
                record(label, "hf_tokenizers", slot.local_size, toks,
                       time.time() - t0, True)
            t0 = time.time()
            slot = train_hf(train_runs, vocab, "bpe")
            toks = count_hf(slot, catchall, held_runs)
            record("hf_bpe", "hf_tokenizers", slot.local_size, toks,
                   time.time() - t0, True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== held-out tokens/word, identical runs (lower is better) ===")
    print(f"{'lang':<6}{'vocab':<8}{'tokenizer':<28}{'kind':<16}{'tok/word':>9}")
    for r in sorted(rows, key=lambda x: (x["lang"], x["vocab"], x["tokens_per_word"])):
        print(f"{r['lang']:<6}{r['vocab']:<8}{r['tokenizer']:<28}"
              f"{r['kind']:<16}{r['tokens_per_word']:>9}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

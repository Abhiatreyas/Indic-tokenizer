"""Diagnose why slot tokenizers fragment words they were trained on.

Hypothesis candidates:
  H1  initial_alphabet=256 eats the budget / distorts Unigram's EM
  H2  training on whitespace-free word runs is too little context
  H3  corpus is too small/repetitive (75 distinct words)
  H4  Unigram pruning (shrinking_factor) collapses the vocab

Run: python diagnose_slot_training.py
"""
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizer.corpora import read_slot_corpus
from tokenizer.train import BYTE_ALPHABET, _BYTE_LEVEL

runs = read_slot_corpus("data/slots/DEVANAGARI.jsonl")
print(f"runs={len(runs)} distinct={len(set(runs))} chars={sum(len(r) for r in runs)}")
print()

PROBE = ["नमस्ते", "भारत", "कंप्यूटर", "विद्यार्थी", "हिंदी", "पानी"]


def build(algo, texts, vocab_size, use_alphabet, **kw):
    if algo == "unigram":
        model = models.Unigram()
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            initial_alphabet=BYTE_ALPHABET if use_alphabet else [],
            special_tokens=["<unk>"],
            unk_token="<unk>",
            show_progress=False,
            **kw,
        )
    else:
        model = models.BPE(unk_token=None)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=BYTE_ALPHABET if use_alphabet else [],
            special_tokens=["<unk>"],
            show_progress=False,
            **kw,
        )
    tok = Tokenizer(model)
    tok.pre_tokenizer = _BYTE_LEVEL
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(list(texts), trainer=trainer)
    return tok


def report(label, tok):
    init = sum(1 for v in tok.get_vocab() if len(v) == 1)
    print(f"{label}")
    print(f"   vocab={tok.get_vocab_size():<6} single-byte entries={init}")
    for w in PROBE:
        ids = tok.encode(w).ids
        back = tok.decode(ids)
        flag = "" if back == w else "  <-- ROUNDTRIP BROKEN"
        print(f"   {w!r:16} {len(ids):>2} tokens{flag}")
    print()


# H1 / H4: alphabet + unigram
report("A) unigram, vocab=1867, alphabet, no shrink limit",
       build("unigram", runs, 1867, True, shrinking_factor=0.999))

report("B) unigram, vocab=1867, alphabet, default shrinking (0.75)",
       build("unigram", runs, 1867, True))

report("C) unigram, vocab=1867, NO alphabet",
       build("unigram", runs, 1867, False, shrinking_factor=0.999))

# H3: more data by repeating
report("D) unigram, vocab=1867, alphabet, runs repeated 20x",
       build("unigram", runs * 20, 1867, True, shrinking_factor=0.999))

# H2 vs BPE
report("E) BPE, vocab=1867, alphabet", build("bpe", runs, 1867, True))
report("F) BPE, vocab=1867, alphabet, runs repeated 20x",
       build("bpe", runs * 20, 1867, True))

# H2: what if runs are joined into longer documents (word groups)?
joined = [" ".join(runs[i:i + 8]) for i in range(0, len(runs), 8)]
report("G) unigram, vocab=1867, alphabet, JOINED runs (8 words/seq)",
       build("unigram", joined, 1867, True, shrinking_factor=0.999))
report("H) BPE, vocab=1867, alphabet, JOINED runs (8 words/seq)",
       build("bpe", joined, 1867, True))

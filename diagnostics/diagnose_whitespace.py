"""Is the LATIN regression caused by whitespace ownership?

Under `neutral` ownership, whitespace belongs to the NEUTRAL slot, so every script
slot is trained on whitespace-free word runs and can never learn a piece spanning
a space. Standard BPE learns " the", " of", "in the". This measures the cost.

Scheme A (neutral): LATIN tokens for the words + 1 NEUTRAL token per space.
Scheme B (leading): one LATIN tokenizer trained on runs that keep their leading
                    space (the SentencePiece U+2581 convention).
"""
from collections import Counter
from tokenizers import Tokenizer, models, decoders, trainers
from tokenizer.corpora import read_slot_corpus, read_documents
from tokenizer.train import BYTE_ALPHABET, _BYTE_LEVEL
from tokenizer.scripts import segment


def train_bpe(texts, vocab_size=765):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = _BYTE_LEVEL
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        list(texts),
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=BYTE_ALPHABET,
            special_tokens=["<unk>"],
            show_progress=False,
        ),
    )
    return tok


docs = [d.text for d in read_documents("data/corpus/documents.jsonl")]
latin_docs = [
    t for t in docs
    if (lambda c: c and c.most_common(1)[0][0] == "LATIN")(
        Counter({r.slot: len(r.text) for r in segment(t) if r.slot != "NEUTRAL"})
    )
]
print(f"latin-dominant documents: {len(latin_docs)}")

# ---- Scheme A: current pipeline (whitespace-free runs) --------------------
runs_a = read_slot_corpus("data/slots/LATIN.jsonl")
print(f"LATIN runs (no whitespace): {len(runs_a)} distinct={len(set(runs_a))}")
tok_a = train_bpe(runs_a)

# ---- Scheme B: leading-space runs ----------------------------------------
runs_b = []
for t in latin_docs:
    buf = ""
    for r in segment(t):
        if r.slot == "LATIN":
            buf += r.text
        elif r.slot == "NEUTRAL" and r.text.strip() == "":
            if buf:
                runs_b.append(buf)
                buf = ""
            # not at start -> becomes the *leading* space of the next run
            buf = r.text
        else:
            if buf:
                runs_b.append(buf)
                buf = ""
    if buf:
        runs_b.append(buf)
runs_b = [r for r in runs_b if r.strip()]
# prepend a space to runs that need one: reconstruct from the doc order
runs_b2 = []
for t in latin_docs:
    cur = ""
    for r in segment(t):
        if r.slot in ("LATIN", "NEUTRAL"):
            cur += r.text
        else:
            if cur:
                runs_b2.append(cur)
                cur = ""
    if cur:
        runs_b2.append(cur)
runs_b2 = [r for r in runs_b2 if r.strip()]
print(f"LATIN runs (leading space): {len(runs_b2)} distinct={len(set(runs_b2))}")
tok_b = train_bpe(runs_b2)

# ---- measure on latin documents -----------------------------------------
a_tokens = 0
b_tokens = 0
chars = 0
for t in latin_docs:
    chars += len(t)
    # A: latin slot tokens + one NEUTRAL token per whitespace run
    for r in segment(t):
        if r.slot == "LATIN":
            a_tokens += len(tok_a.encode(r.text).ids)
        elif r.slot == "NEUTRAL" and r.text.strip() == "":
            a_tokens += 1
    # B: one tokenizer over the whole document (spaces included)
    b_tokens += len(tok_b.encode(t).ids)

print()
print(f"chars                       {chars}")
print(f"A (neutral)   tokens        {a_tokens:<8} tokens/char={a_tokens/chars:.4f}")
print(f"B (leading)   tokens        {b_tokens:<8} tokens/char={b_tokens/chars:.4f}")
print(f"improvement of B over A     {100*(a_tokens-b_tokens)/a_tokens:.2f}%")

print()
for s in [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require large amounts of data.",
]:
    print(f"{s!r}")
    print(f"   A: {[tok_a.decode([i]) for i in tok_a.encode(s).ids]}")
    print(f"   B: {[tok_b.decode([i]) for i in tok_b.encode(s).ids]}")

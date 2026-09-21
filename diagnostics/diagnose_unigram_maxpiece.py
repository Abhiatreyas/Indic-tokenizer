"""Pin the Unigram failure down to a specific parameter.

HF UnigramTrainer defaults to max_piece_length=16 *pre-tokenized characters*.
Under ByteLevel, one Devanagari letter is 3 such characters, so a 6-letter word is
18 characters and is structurally unrepresentable as a single piece.
"""
from tokenizers import Tokenizer, models, decoders, trainers
from tokenizer.corpora import read_slot_corpus
from tokenizer.train import BYTE_ALPHABET, _BYTE_LEVEL

runs = read_slot_corpus("data/slots/DEVANAGARI.jsonl")
PROBE = ["नमस्ते", "भारत", "कंप्यूटर", "विद्यार्थी", "हिंदी", "पानी"]

print("byte lengths of probes (ByteLevel chars = UTF-8 bytes):")
for w in PROBE:
    print(f"   {w!r:16} {len(w)} devanagari chars -> {len(w.encode('utf-8'))} byte-chars")
print()


def uni(max_piece_length, vocab_size=1867):
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = _BYTE_LEVEL
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        list(runs),
        trainer=trainers.UnigramTrainer(
            vocab_size=vocab_size,
            initial_alphabet=BYTE_ALPHABET,
            special_tokens=["<unk>"],
            unk_token="<unk>",
            show_progress=False,
            max_piece_length=max_piece_length,
        ),
    )
    return tok


for mpl in (16, 24, 32, 64):
    tok = uni(mpl)
    lens = [len(tok.encode(w).ids) for w in PROBE]
    print(f"max_piece_length={mpl:<3} vocab={tok.get_vocab_size():<5} "
          f"probe token counts={lens}")

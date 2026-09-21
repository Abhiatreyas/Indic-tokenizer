"""Per-slot tokenizer training.

Every slot is trained with a ByteLevel pre-tokenizer seeded with the full
256-character byte alphabet. That seeding is what makes the round-trip invariant
hold (SPEC.md 5, 8.1): the model can always fall back to emitting individual byte
characters, so no input is ever unrepresentable and the unknown rate is zero by
construction.

Special tokens are deliberately kept OUT of each slot's content id space.
HuggingFace `tokenizers` gives added/special tokens the lowest local ids, so if
they were left in, local id 0 of every slot would be "<pad>" and the global
special range would be shadowed by per-slot copies. Instead:
  * the trainers get exactly one special, "<unk>", which UnigramTrainer requires;
  * it is then stripped from the content vocab;
  * global specials live only in the merged layout, at ids 0..k-1.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .slots import BYTE_TABLE_SIZE

# add_prefix_space=False is essential: with it True, decoding inserts a leading
# space and round-trip fails. use_regex=False avoids GPT-2's regex splitting so
# the mapping is a pure, order-preserving byte encoding.
_BYTE_LEVEL = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
BYTE_ALPHABET: List[str] = list(pre_tokenizers.ByteLevel.alphabet())

# Only special any slot trainer needs. Global specials are a merge-level concept.
TRAIN_SPECIALS: Sequence[str] = ("<unk>",)

# How much of the byte alphabet each trained slot is seeded with.
#   "observed"        -- only the bytes that occur in that slot's corpus
#                        (typically 40-60). Cheapest, but any character outside
#                        the corpus -- a newline, a digit, a quote -- triggers
#                        fallback for whatever span contains it.
#   "observed+ascii"  -- observed bytes plus the 98 whitespace/printable-ASCII
#                        bytes. Costs ~40 extra rows per slot and removes the
#                        overwhelmingly common out-of-corpus cases. Default.
#   "full"            -- all 256 bytes in every slot (~1311 wasted rows).
# Losslessness holds under all three: anything a slot cannot represent is
# isolated and re-encoded through the shared CATCHALL byte table.
ALPHABET_OBSERVED = "observed"
ALPHABET_OBSERVED_ASCII = "observed+ascii"
ALPHABET_FULL = "full"
ALPHABET_SCOPES = (ALPHABET_OBSERVED, ALPHABET_OBSERVED_ASCII, ALPHABET_FULL)

# Tab, LF, CR, and printable ASCII. The characters most likely to appear in real
# text without appearing in any particular slot's corpus.
_ASCII_SAFETY = frozenset([0x09, 0x0A, 0x0D] + list(range(0x20, 0x7F)))


def observed_alphabet(
    texts: Sequence[str], include_ascii: bool = True
) -> List[str]:
    """ByteLevel characters for the bytes present in `texts`, plus ASCII safety.

    Returns a list in byte order; it is used as `initial_alphabet`, so a shorter
    list directly reclaims vocabulary rows.
    """
    seen = bytearray(256)
    for t in texts:
        for b in t.encode("utf-8"):
            seen[b] = 1
    if include_ascii:
        for b in _ASCII_SAFETY:
            seen[b] = 1
    return [BYTE_ALPHABET[b] for b in range(256) if seen[b]]


class UnrepresentableRun(Exception):
    """A slot model produced an id outside its content vocabulary.

    The partitioner catches this and re-encodes the run with the CATCHALL byte
    table, so losslessness never depends on the trained model behaving.
    """


def _trainer_for(
    algorithm: str,
    vocab_size: int,
    specials: Sequence[str],
    alphabet: Sequence[str],
    max_piece_length: int = 48,
):
    common = dict(
        vocab_size=vocab_size,
        initial_alphabet=list(alphabet),
        special_tokens=list(specials),
        show_progress=False,
    )
    if algorithm == "bpe":
        return trainers.BpeTrainer(**common)
    if algorithm == "unigram":
        # max_piece_length counts PRE-TOKENIZED tokens, so under ByteLevel it is
        # a byte budget, not a character budget. The default of 16 caps a piece
        # at ~5 Devanagari/Kannada aksharas (3 bytes each) while SentencePiece's
        # equivalent limit of 16 counts Unicode characters. Measured on real
        # Indic text at 32k vocab this single setting is worth 22%:
        # Kannada 1.9210 -> 1.4965 tokens/word, Hindi 1.2983 -> 1.0589.
        # 48 is the knee of the curve; 64 and 128 add nothing.
        return trainers.UnigramTrainer(
            unk_token="<unk>", max_piece_length=max_piece_length, **common
        )
    raise ValueError(f"unknown algorithm {algorithm!r}")


def _model_for(algorithm: str):
    if algorithm == "bpe":
        # unk_token MUST be set. With unk_token=None, HF's BPE does not merely
        # avoid emitting <unk> -- it silently DROPS characters outside the
        # alphabet, producing no id at all, so the loss is undetectable from the
        # id stream and round-trip fails quietly. With <unk> set, an
        # out-of-alphabet character becomes a special id, which
        # HFTokenizerSlot.encode turns into UnrepresentableRun, which the
        # partitioner turns into a lossless CATCHALL byte fallback.
        return models.BPE(unk_token="<unk>")
    if algorithm == "unigram":
        return models.Unigram()
    raise ValueError(f"unknown algorithm {algorithm!r}")


def train_slot(
    slot: str,
    algorithm: str,
    texts: Sequence[str],
    vocab_size: int,
    specials: Sequence[str] = TRAIN_SPECIALS,
    alphabet_scope: str = ALPHABET_OBSERVED_ASCII,
    max_piece_length: int = 48,
) -> "HFTokenizerSlot":
    """Train one slot's tokenizer on that slot's runs."""
    if not texts:
        raise ValueError(f"slot {slot!r} has no training text")
    if alphabet_scope not in ALPHABET_SCOPES:
        raise ValueError(
            f"unknown alphabet_scope {alphabet_scope!r}; expected {ALPHABET_SCOPES}"
        )

    if alphabet_scope == ALPHABET_FULL:
        alphabet = BYTE_ALPHABET
    else:
        alphabet = observed_alphabet(
            texts, include_ascii=(alphabet_scope == ALPHABET_OBSERVED_ASCII)
        )

    tok = Tokenizer(_model_for(algorithm))
    tok.pre_tokenizer = _BYTE_LEVEL
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        list(texts),
        trainer=_trainer_for(
            algorithm, vocab_size, specials, alphabet, max_piece_length
        ),
    )
    return HFTokenizerSlot(slot, algorithm, tok)


class HFTokenizerSlot:
    """A trained slot, exposing a content-only local id space."""

    def __init__(self, slot: str, algorithm: str, tokenizer: Tokenizer) -> None:
        self.slot = slot
        self.algorithm = algorithm
        self.tokenizer = tokenizer

        vocab = tokenizer.get_vocab()  # token string -> hf id
        special_ids = set(tokenizer.get_added_tokens_decoder().keys())

        ordered = sorted(vocab.items(), key=lambda kv: kv[1])
        self._content: List[str] = [
            tok for tok, i in ordered if i not in special_ids
        ]
        self.local_size = len(self._content)

        self._hf_to_local: Dict[int, int] = {}
        self._local_to_hf: List[int] = [0] * self.local_size
        for local_id, (tok, hf_id) in enumerate(
            (t, i) for t, i in ordered if i not in special_ids
        ):
            self._hf_to_local[hf_id] = local_id
            self._local_to_hf[local_id] = hf_id

    # --- interface used by the partitioner --------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode to content-local ids.

        Raises UnrepresentableRun if the model emits anything outside its content
        vocabulary, rather than silently dropping or corrupting it.
        """
        hf_ids = self.tokenizer.encode(text).ids
        out: List[int] = []
        for hf in hf_ids:
            local = self._hf_to_local.get(hf)
            if local is None:
                raise UnrepresentableRun(
                    f"slot {self.slot!r} emitted non-content hf id {hf}"
                )
            out.append(local)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        if not ids:
            return ""
        hf_ids = []
        for local in ids:
            if not 0 <= local < self.local_size:
                raise ValueError(
                    f"slot {self.slot!r} local id {local} out of range "
                    f"(0..{self.local_size - 1})"
                )
            hf_ids.append(self._local_to_hf[local])
        return self.tokenizer.decode(hf_ids, skip_special_tokens=False)

    def local_vocab(self) -> List[str]:
        return list(self._content)

    def save(self, path: str) -> None:
        self.tokenizer.save(path)


class ByteFallbackSlot:
    """CATCHALL: a fixed 256-entry byte table, not a trained model.

    Local id == byte value. This is the escape hatch that keeps unmodelled
    scripts (Tamil, Arabic, CJK, unusual emoji sequences) lossless instead of
    turning them into <unk>.
    """

    slot = "CATCHALL"
    algorithm = "bytes"
    local_size = BYTE_TABLE_SIZE

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(ids).decode("utf-8", errors="strict")

    def local_vocab(self) -> List[str]:
        # Same representation the other slots use for raw bytes.
        return list(BYTE_ALPHABET)

    def save(self, path: str) -> None:  # nothing to persist; fully specified
        return None


def load_hf_slot(slot: str, algorithm: str, path: str) -> HFTokenizerSlot:
    return HFTokenizerSlot(slot, algorithm, Tokenizer.from_file(path))

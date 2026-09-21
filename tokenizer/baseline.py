"""The shared-vocab baseline, and the comparison that decides the design.

The baseline is one tokenizer over a script-balanced mix of the same corpora,
sized to the same total vocabulary as the partitioned build. Same embedding
parameter count on both sides, so the comparison is about vocabulary *structure*
and nothing else (SPEC.md 10.1).

Per-script comparison uses script-homogeneous document groups. Measuring a
shared-vocab model on whitespace-stripped runs would disadvantage it unfairly,
since it was trained on full documents; grouping whole documents by dominant
script gives both tokenizers their natural input.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .merge import DEFAULT_SPECIALS
from .metrics import embedding_params
from .scripts import NEUTRAL, segment
from .train import BYTE_ALPHABET, _BYTE_LEVEL


class BaselineTokenizer:
    """A single shared vocabulary over all scripts."""

    def __init__(self, tokenizer: Tokenizer, specials: Sequence[str]) -> None:
        self.tokenizer = tokenizer
        self.specials = list(specials)
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        vocab_size: int,
        specials: Sequence[str] = DEFAULT_SPECIALS,
        algorithm: str = "unigram",
    ) -> "BaselineTokenizer":
        if algorithm == "unigram":
            model = models.Unigram()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size,
                initial_alphabet=BYTE_ALPHABET,
                special_tokens=list(specials),
                unk_token="<unk>",
                show_progress=False,
            )
        elif algorithm == "bpe":
            model = models.BPE(unk_token=None)
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                initial_alphabet=BYTE_ALPHABET,
                special_tokens=list(specials),
                show_progress=False,
            )
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")

        tok = Tokenizer(model)
        tok.pre_tokenizer = _BYTE_LEVEL
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(list(texts), trainer=trainer)
        return cls(tok, specials)

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    @classmethod
    def from_file(cls, path: str, specials: Sequence[str] = DEFAULT_SPECIALS):
        return cls(Tokenizer.from_file(path), specials)

    def save(self, path: str) -> None:
        self.tokenizer.save(path)


# --------------------------------------------------------------------------
# Script balancing
# --------------------------------------------------------------------------


def dominant_slot(text: str) -> Tuple[str, int]:
    """The non-neutral slot holding the most characters in `text`."""
    counts: Counter = Counter()
    for run in segment(text):
        if run.slot != NEUTRAL:
            counts[run.slot] += len(run.text)
    if not counts:
        return NEUTRAL, 0
    slot, n = counts.most_common(1)[0]
    total = sum(counts.values())
    return slot, int(100 * n / total) if total else 0


def balance_documents(
    docs: Sequence, seed: int = 1337, min_purity: int = 90
) -> List[str]:
    """Equalise byte contribution per script so Latin does not dominate.

    Keeps only documents that are >= `min_purity` percent one non-neutral slot,
    then truncates every script group to the smallest group's byte total.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for d in docs:
        slot, purity = dominant_slot(d.text)
        if slot == NEUTRAL or purity < min_purity:
            continue
        groups[slot].append(d.text)

    if not groups:
        return [d.text for d in docs]

    by_bytes = {
        s: sum(len(t.encode("utf-8")) for t in ts) for s, ts in groups.items()
    }
    target = min(by_bytes.values())

    rng = random.Random(seed)
    out: List[str] = []
    for slot, texts in sorted(groups.items()):
        rng.shuffle(texts)
        acc = 0
        for t in texts:
            if acc >= target:
                break
            out.append(t)
            acc += len(t.encode("utf-8"))
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _aggregate(encode, texts: Sequence[str]) -> Dict[str, float]:
    tokens = 0
    chars = 0
    byts = 0
    for t in texts:
        tokens += len(encode(t))
        chars += len(t)
        byts += len(t.encode("utf-8"))
    return {
        "documents": len(texts),
        "tokens": tokens,
        "chars": chars,
        "bytes": byts,
        "tokens_per_char": round(tokens / chars, 4) if chars else None,
        "bytes_per_token": round(byts / tokens, 4) if tokens else None,
    }


def compare(
    partitioned,
    baseline: BaselineTokenizer,
    docs: Sequence,
    d_model: int = 2048,
) -> Dict:
    """Measure both tokenizers on the same script-homogeneous document groups."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for d in docs:
        slot, purity = dominant_slot(d.text)
        if slot == NEUTRAL or purity < 90:
            continue
        groups[slot].append(d.text)

    per_script: Dict[str, dict] = {}
    for slot, texts in sorted(groups.items()):
        p = _aggregate(partitioned.encode, texts)
        b = _aggregate(baseline.encode, texts)
        p_tpc = p["tokens_per_char"]
        b_tpc = b["tokens_per_char"]
        per_script[slot] = {
            "documents": len(texts),
            "chars": p["chars"],
            "partitioned": p,
            "baseline": b,
            "tokens_per_char_delta": (
                round(p_tpc - b_tpc, 4) if p_tpc and b_tpc else None
            ),
            "tokens_per_char_improvement_pct": (
                round(100 * (b_tpc - p_tpc) / b_tpc, 2) if p_tpc and b_tpc else None
            ),
            "partitioned_wins": (p_tpc < b_tpc) if p_tpc and b_tpc else None,
        }

    all_texts = [t for ts in groups.values() for t in ts]
    p_all = _aggregate(partitioned.encode, all_texts)
    b_all = _aggregate(baseline.encode, all_texts)

    pv = partitioned.layout.vocab_size
    bv = baseline.vocab_size

    return {
        "per_script": per_script,
        "overall_partitioned": p_all,
        "overall_baseline": b_all,
        "vocab": {
            "partitioned": pv,
            "baseline": bv,
            "partitioned_embedding_params": embedding_params(pv, d_model),
            "baseline_embedding_params": embedding_params(bv, d_model),
        },
        "gates": {
            "G4_indic_fertility_better": all(
                per_script[s]["partitioned_wins"]
                for s in ("DEVANAGARI", "KANNADA")
                if s in per_script
            )
            if any(s in per_script for s in ("DEVANAGARI", "KANNADA"))
            else None,
            "G5_no_script_regression": all(
                (v["tokens_per_char_improvement_pct"] or 0) >= -1.0
                for v in per_script.values()
            ),
        },
    }

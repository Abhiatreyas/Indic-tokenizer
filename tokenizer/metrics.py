"""Measurement.

The decider for the partitioned-vs-shared comparison is bits-per-byte on held-out
text, because it is comparable across tokenizers with different vocab sizes
(unlike per-token loss). Fertility is reported per slot as tokens-per-character
rather than tokens-per-word: Indic words are long and agglutinative, so
whitespace-delimited word counts are not comparable across scripts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

from .scripts import segment


def per_slot_metrics(tokenizer, texts: Iterable[str]) -> Dict[str, dict]:
    """Tokens/chars/bytes per slot, measured on the paths encoding takes."""
    agg: Dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for run, ids in tokenizer.encode_runs(text):
            a = agg[run.slot]
            a["runs"] += 1
            a["chars"] += len(run.text)
            a["bytes"] += len(run.text.encode("utf-8"))
            a["tokens"] += len(ids)

    out: Dict[str, dict] = {}
    for slot, a in sorted(agg.items()):
        chars = a["chars"] or 1
        toks = a["tokens"]
        out[slot] = {
            "runs": a["runs"],
            "chars": a["chars"],
            "bytes": a["bytes"],
            "tokens": toks,
            "tokens_per_char": round(toks / chars, 4),
            "chars_per_token": round(chars / toks, 4) if toks else None,
            "bytes_per_token": (
                round(a["bytes"] / toks, 4) if toks else None
            ),
        }
    return out


def corpus_metrics(tokenizer, texts: Sequence[str]) -> dict:
    """Whole-corpus compression and fertility."""
    total_tokens = 0
    total_bytes = 0
    total_chars = 0
    total_words = 0
    for text in texts:
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
        total_bytes += len(text.encode("utf-8"))
        total_chars += len(text)
        total_words += len(text.split())

    return {
        "documents": len(texts),
        "tokens": total_tokens,
        "bytes": total_bytes,
        "chars": total_chars,
        "words": total_words,
        "tokens_per_word": round(total_tokens / total_words, 4) if total_words else None,
        "bytes_per_token": round(total_bytes / total_tokens, 4) if total_tokens else None,
        "tokens_per_char": round(total_tokens / total_chars, 4) if total_chars else None,
    }


def roundtrip_report(
    tokenizer, texts: Sequence[str], normalize_form: str = "NFC"
) -> dict:
    """Gate G1: exact reconstruction, measured, not assumed."""
    from .scripts import normalize

    failures: List[dict] = []
    total_ids = 0
    special_hits = 0
    # An opt-in slot marker is a structural special, not an unknown token: it is
    # expected in the stream and is skipped on decode.
    slot_sep = getattr(tokenizer, "slot_sep_id", None)
    for text in texts:
        ids = tokenizer.encode(text)
        back = tokenizer.decode(ids)
        expected = normalize(text, normalize_form)
        total_ids += len(ids)
        special_hits += sum(
            1
            for i in ids
            if i < tokenizer.layout.special_end and i != slot_sep
        )
        if back != expected:
            if len(failures) < 5:
                failures.append(
                    {
                        "in": text[:120],
                        "out": back[:120],
                        "expected": expected[:120],
                    }
                )
            else:
                failures.append({"in": text[:60], "out": "<omitted>"})

    return {
        "documents": len(texts),
        "exact": len(texts) - len(failures),
        "failures": len(failures),
        "exactness": round((len(texts) - len(failures)) / len(texts), 6)
        if texts
        else 1.0,
        "total_ids": total_ids,
        "special_ids_in_output": special_hits,
        "examples": failures[:5],
    }


def vocab_utilization(tokenizer, texts: Sequence[str]) -> dict:
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenizer.encode(text))

    vocab_size = tokenizer.layout.vocab_size
    used = len(counts)
    per_slot: Dict[str, dict] = {}
    for rng in tokenizer.layout.slots:
        in_slot = sum(1 for i in counts if rng.start <= i < rng.end)
        per_slot[rng.name] = {
            "size": rng.size,
            "used": in_slot,
            "utilization": round(in_slot / rng.size, 4) if rng.size else None,
        }

    return {
        "vocab_size": vocab_size,
        "used": used,
        "utilization": round(used / vocab_size, 4) if vocab_size else None,
        "per_slot": per_slot,
    }


def embedding_params(vocab_size: int, d_model: int, tied: bool = True) -> int:
    """Embedding-table parameter count. The cost side of the vocab budget."""
    params = vocab_size * d_model
    if not tied:
        params *= 2
    return params

"""Slot specifications, budget allocation, and the frozen id layout.

A slot is declared as ``(name, algorithm, scripts[])``. That one shape covers
every grouping the design needs:

  * several languages sharing a script share a slot (LATIN holds English,
    Hinglish, Kanglish, French, ...; DEVANAGARI holds Hindi, Marathi, Nepali,
    Sanskrit);
  * different slots may run different algorithms;
  * several scripts may be grouped into one slot, which is the right call for
    low-resource scripts that cannot each afford a dedicated vocabulary.

Grouping by algorithm *alone* -- all BPE scripts into one slot -- is the shared
vocabulary baseline, which measured 13% worse on Devanagari and 36% worse on
Kannada. The comparison harness exists to keep that honest.

On ids: there is no magic sentinel. ``first_invalid_id`` is derived from the
layout, an in-stream marker is an ordinary special token at a low id, and the
only reserved tail is embedding padding to a hardware-friendly multiple.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from .scripts import (
    CATCHALL,
    CODE_MATH,
    COMMON,
    DEFAULT_SLOT_MAP,
    DEVANAGARI,
    KANNADA,
    LATIN,
    NEUTRAL,
    UNKNOWN,
)

# CATCHALL is not trained; it is a fixed 256-entry byte table.
BYTE_TABLE_SIZE = 256

# The only sentinel the design may ever need. It is an ordinary special token at
# a low id, NOT a magic number: adding it to `specials` costs exactly one row.
# Off by default, because the chosen stream format is markerless.
SLOT_SEP = "<|slot_sep|>"

# Valid slot algorithms.
ALGORITHMS = ("bpe", "unigram", "bytes")

# Embedding rows are conventionally padded to a multiple of this for
# tensor-parallel sharding. Padding is recorded but never emitted.
DEFAULT_EMBED_MULTIPLE = 128

# Sentinel used inside the flat lookup table.
_NO_SLOT = 0xFF


@dataclass(frozen=True)
class SlotSpec:
    """One slot: an algorithm applied to a set of scripts.

    `scripts` is the routing key, and it is data. Adding a language to a slot is
    a config edit, never a change to detection code.
    """

    name: str
    algorithm: str  # "bpe" | "unigram" | "bytes"
    scripts: tuple[str, ...] = ()
    # Code/math spans route here regardless of script (SPEC.md 7).
    structural: bool = False
    trained: bool = True
    # CATCHALL is fixed-size and excluded from budget sharing.
    exempt_from_budget: bool = False
    # Unigram only. Counts PRE-TOKENIZED tokens (bytes under ByteLevel), so
    # Indic scripts need ~3x the character budget. Ignored by BPE.
    max_piece_length: int = 48

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "scripts": list(self.scripts),
            "structural": self.structural,
            "trained": self.trained,
            "exempt_from_budget": self.exempt_from_budget,
            "max_piece_length": self.max_piece_length,
        }


def _default_specs() -> tuple[SlotSpec, ...]:
    """Specs derived from the canonical script->slot map, so they cannot drift."""
    by_slot: Dict[str, List[str]] = {}
    for script, slot in DEFAULT_SLOT_MAP.items():
        by_slot.setdefault(slot, []).append(script)

    specs: List[SlotSpec] = []
    for name in (NEUTRAL, LATIN, DEVANAGARI, KANNADA):
        # LATIN and the Indic slots use Unigram with a raised piece limit.
        # Measured on pooled real slot content (see diagnostics/README.md 14):
        #   LATIN holding English + Hinglish + Tanglish
        #     english 1.4294 -> 1.3155, hinglish 1.0322 -> 0.9320,
        #     tanglish 1.7404 -> 1.6476  (Unigram wins all three)
        #   DEVANAGARI shared across Hindi + Marathi + Nepali
        #     +5.5% / +7.8% / +7.1%
        # Per-language slots at 16k are a coin flip, but the design SHARES a slot
        # across the languages of a script, and that is the case Unigram wins.
        # NEUTRAL and CODE_MATH stay BPE: no benchmark covers them and BPE is the
        # conventional choice for punctuation and code.
        algo = "bpe" if name == NEUTRAL else "unigram"
        specs.append(
            SlotSpec(name, algo, tuple(sorted(by_slot[name])), max_piece_length=48)
        )
    specs.append(SlotSpec(CODE_MATH, "bpe", (), structural=True))
    specs.append(
        SlotSpec(CATCHALL, "bytes", (), trained=False, exempt_from_budget=True)
    )
    return tuple(specs)


DEFAULT_SPECS: Sequence[SlotSpec] = _default_specs()


def slot_map_from_specs(specs: Sequence[SlotSpec]) -> Dict[str, str]:
    """script -> slot name, derived from the specs.

    The bytes slot is excluded: it is the fallback for unmapped scripts, not a
    script owner.
    """
    out: Dict[str, str] = {}
    for spec in specs:
        if spec.algorithm == "bytes":
            continue
        for script in spec.scripts:
            if script in out:
                raise ValueError(
                    f"script {script!r} claimed by both {out[script]!r} and "
                    f"{spec.name!r}; a script may belong to exactly one slot"
                )
            out[script] = spec.name
    return out


def catchall_spec(specs: Sequence[SlotSpec]) -> SlotSpec:
    for spec in specs:
        if spec.algorithm == "bytes":
            return spec
    raise ValueError("specs define no bytes slot; unmapped scripts would be lost")


def validate_specs(specs: Sequence[SlotSpec]) -> None:
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate slot names: {names}")
    for spec in specs:
        if spec.algorithm not in ALGORITHMS:
            raise ValueError(f"slot {spec.name!r}: unknown algorithm {spec.algorithm!r}")
        if spec.algorithm == "bytes" and spec.trained:
            raise ValueError(f"slot {spec.name!r}: bytes slots are not trained")
    structural = [s.name for s in specs if s.structural]
    if len(structural) > 1:
        raise ValueError(f"more than one structural slot: {structural}")
    catchall_spec(specs)  # raises if absent
    slot_map_from_specs(specs)  # raises on double-claimed scripts


def specs_from_dicts(items: Sequence[Mapping]) -> tuple[SlotSpec, ...]:
    """Build specs from config (see config/slots.v1.yaml)."""
    specs = tuple(
        SlotSpec(
            name=it["name"],
            algorithm=it.get("algorithm", "bpe"),
            scripts=tuple(it.get("scripts", ())),
            structural=bool(it.get("structural", False)),
            trained=bool(it.get("trained", True)),
            exempt_from_budget=bool(it.get("exempt_from_budget", False)),
            max_piece_length=int(it.get("max_piece_length", 48)),
        )
        for it in items
    )
    validate_specs(specs)
    return specs


# --------------------------------------------------------------------------
# Id layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotRange:
    name: str
    start: int
    end: int  # exclusive
    algorithm: str
    size: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "algorithm": self.algorithm,
            "size": self.size,
        }


@dataclass
class IdLayout:
    """The frozen id layout: specials, then slots in declared order.

    Ids are contiguous and dense. There is no sentinel id: `first_invalid_id` is
    simply the end of the allocated range.
    """

    specials: List[str]
    slots: List[SlotRange]
    embed_multiple: int = DEFAULT_EMBED_MULTIPLE

    @classmethod
    def from_sizes(
        cls,
        specials: Sequence[str],
        slot_sizes: Mapping[str, int],
        specs: Sequence[SlotSpec] = DEFAULT_SPECS,
        embed_multiple: int = DEFAULT_EMBED_MULTIPLE,
    ) -> "IdLayout":
        specials = list(specials)
        # Slot ids begin after the special-token range, which owns 0..k-1.
        cursor = len(specials)
        ranges: List[SlotRange] = []
        for spec in specs:
            if spec.name not in slot_sizes:
                raise ValueError(f"missing size for slot {spec.name!r}")
            size = int(slot_sizes[spec.name])
            if size <= 0:
                raise ValueError(f"slot {spec.name!r} has non-positive size {size}")
            ranges.append(
                SlotRange(spec.name, cursor, cursor + size, spec.algorithm, size)
            )
            cursor += size

        layout = cls(specials=specials, slots=ranges, embed_multiple=embed_multiple)
        layout.validate()
        return layout

    # --- queries ----------------------------------------------------------

    @property
    def special_end(self) -> int:
        return len(self.specials)

    @property
    def vocab_size(self) -> int:
        """Allocated rows. The first invalid id equals this."""
        return self.slots[-1].end

    @property
    def first_invalid_id(self) -> int:
        """The only bound the decoder needs. Derived, not declared."""
        return self.vocab_size

    @property
    def padded_vocab_size(self) -> int:
        """vocab_size rounded up for tensor-parallel embedding sharding.

        Padding rows are allocated by the model but never emitted, so they carry
        no meaning and need no token string.
        """
        if self.embed_multiple <= 1:
            return self.vocab_size
        m = self.embed_multiple
        return ((self.vocab_size + m - 1) // m) * m

    @property
    def boundaries(self) -> List[int]:
        return [s.start for s in self.slots]

    def slot_of(self, token_id: int) -> Optional[SlotRange]:
        """Binary-search fallback. Prefer SlotLookup for hot paths."""
        if token_id < self.special_end or token_id >= self.vocab_size:
            return None
        lo, hi = 0, len(self.slots) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s = self.slots[mid]
            if token_id < s.start:
                hi = mid - 1
            elif token_id >= s.end:
                lo = mid + 1
            else:
                return s
        return None

    def slot_by_name(self, name: str) -> SlotRange:
        for s in self.slots:
            if s.name == name:
                return s
        raise KeyError(name)

    def lookup(self) -> "SlotLookup":
        return SlotLookup(self)

    # --- validation -------------------------------------------------------

    def validate(self) -> None:
        if len(set(self.specials)) != len(self.specials):
            raise ValueError("duplicate special tokens")

        expected = self.special_end
        for s in self.slots:
            if s.start != expected:
                raise ValueError(
                    f"slot {s.name!r} starts at {s.start}, expected {expected} "
                    "(slots must be contiguous and ordered)"
                )
            if s.end <= s.start:
                raise ValueError(f"slot {s.name!r} is empty")
            expected = s.end

        if self.embed_multiple < 1:
            raise ValueError("embed_multiple must be >= 1")
        if len(self.slots) > _NO_SLOT:
            raise ValueError("too many slots for the flat lookup table")

    # --- serialization ----------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "special_tokens": list(self.specials),
            "vocab_size": self.vocab_size,
            "first_invalid_id": self.first_invalid_id,
            "embed_multiple": self.embed_multiple,
            "padded_vocab_size": self.padded_vocab_size,
            "slots": [s.as_dict() for s in self.slots],
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    @classmethod
    def from_dict(cls, d: Mapping) -> "IdLayout":
        slots = [
            SlotRange(
                name=s["name"],
                start=int(s["start"]),
                end=int(s["end"]),
                algorithm=s["algorithm"],
                size=int(s["size"]),
            )
            for s in d["slots"]
        ]
        layout = cls(
            specials=list(d["special_tokens"]),
            slots=slots,
            embed_multiple=int(d.get("embed_multiple", DEFAULT_EMBED_MULTIPLE)),
        )
        layout.validate()
        return layout

    @classmethod
    def from_json(cls, path: str) -> "IdLayout":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


class SlotLookup:
    """O(1) id -> slot routing.

    One byte per allocated id: a flat table is cheaper than a binary search on
    every id and trivially auditable. At a 260k vocab this is 260 KB.
    """

    __slots__ = ("_table", "_slots", "_special_end", "_vocab_size")

    def __init__(self, layout: IdLayout) -> None:
        if len(layout.slots) > _NO_SLOT:
            raise ValueError("too many slots for a flat lookup table")
        table = bytearray([_NO_SLOT]) * layout.vocab_size
        for i, s in enumerate(layout.slots):
            table[s.start : s.end] = bytes([i]) * s.size
        self._table = table
        self._slots = layout.slots
        self._special_end = layout.special_end
        self._vocab_size = layout.vocab_size

    def slot_of(self, token_id: int) -> Optional[SlotRange]:
        if token_id < self._special_end or token_id >= self._vocab_size:
            return None
        idx = self._table[token_id]
        if idx == _NO_SLOT:
            return None
        return self._slots[idx]

    def slot_name(self, token_id: int) -> Optional[str]:
        s = self.slot_of(token_id)
        return None if s is None else s.name


# --------------------------------------------------------------------------
# Budget allocation
# --------------------------------------------------------------------------


def allocate_budgets(
    slot_bytes: Mapping[str, int],
    total: int,
    specs: Sequence[SlotSpec] = DEFAULT_SPECS,
    alpha: float = 0.7,
    min_slot: int = 300,
    max_slot: int = 64000,
) -> Dict[str, int]:
    """Split a total vocab budget across trained slots.

    `alpha` tempers the allocation. alpha = 1.0 is literal proportionality to
    corpus bytes, which starves small scripts because Latin-script corpora
    dominate raw byte counts -- the exact failure this tokenizer exists to
    prevent. alpha = 0.7 is the draft default (SPEC.md 6.1).

    Water-filling: clamp to [min_slot, max_slot], then redistribute the excess
    among the unclamped slots until stable.
    """
    exempt = {s.name for s in specs if s.exempt_from_budget}
    trained = [s.name for s in specs if not s.exempt_from_budget]

    remaining = total - len(exempt) * BYTE_TABLE_SIZE
    if remaining <= 0:
        raise ValueError("total vocab too small once exempt slots are reserved")

    weights = {n: float(slot_bytes.get(n, 0)) ** alpha for n in trained}
    if sum(weights.values()) <= 0:
        weights = {n: 1.0 for n in trained}

    budgets: Dict[str, int] = {}
    free = list(trained)
    pool = float(remaining)

    while free:
        wsum = sum(weights[n] for n in free)
        if wsum <= 0:
            proposed = {n: pool / len(free) for n in free}
        else:
            proposed = {n: pool * weights[n] / wsum for n in free}

        clamped = False
        for n in list(free):
            if proposed[n] < min_slot:
                budgets[n] = min_slot
                pool -= min_slot
                free.remove(n)
                clamped = True
            elif proposed[n] > max_slot:
                budgets[n] = max_slot
                pool -= max_slot
                free.remove(n)
                clamped = True
        if not clamped:
            for n in free:
                budgets[n] = int(round(proposed[n]))
            break

    for name in exempt:
        budgets[name] = BYTE_TABLE_SIZE

    drift = total - sum(budgets.values())
    if drift != 0:
        adjustable = [n for n in trained if min_slot <= budgets[n] + drift <= max_slot]
        if adjustable:
            target = max(adjustable, key=lambda n: budgets[n])
            budgets[target] += drift
    return budgets

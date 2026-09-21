"""The partitioned tokenizer: encoder and decoder.

Encode preserves original order and emits no markers. Decode recovers each id's
slot from a flat lookup table and dispatches to that slot's algorithm
(SPEC.md sections 7 and 8).

Losslessness does not depend on any trained model behaving well: if a slot model
emits something outside its content vocabulary, the run is re-encoded with the
CATCHALL byte table instead. That keeps the round-trip invariant a property of
the design rather than a hope about BPE/Unigram.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .merge import MODELS_DIRNAME, SLOTS_NAME, VOCAB_NAME, load_specs
from .scripts import (
    CATCHALL,
    CODE_MATH,
    DEFAULT_SLOT_MAP,
    DEFAULT_WHITESPACE_OWNERSHIP,
    WHITESPACE_MODES,
    Run,
    segment,
)
from .slots import (
    DEFAULT_SPECS,
    SLOT_SEP,
    IdLayout,
    SlotLookup,
    SlotSpec,
    slot_map_from_specs,
)
from .scripts import WHITESPACE_MODES

from .train import ByteFallbackSlot, UnrepresentableRun, load_hf_slot


@dataclass
class EncodeStats:
    fallback_runs: int = 0
    fallback_chars: int = 0
    runs: int = 0
    ids: int = 0
    slot_runs: Counter = field(default_factory=Counter)
    slot_tokens: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "runs": self.runs,
            "ids": self.ids,
            "fallback_runs": self.fallback_runs,
            "fallback_chars": self.fallback_chars,
            "slot_runs": dict(self.slot_runs),
            "slot_tokens": dict(self.slot_tokens),
        }


class PartitionedTokenizer:
    def __init__(
        self,
        layout: IdLayout,
        slots: Dict[str, object],
        vocab: Sequence[str],
        specs: Sequence[SlotSpec] = DEFAULT_SPECS,
        tokenizer_version: str = "v1",
        emit_slot_markers: bool = False,
        whitespace_ownership: str = DEFAULT_WHITESPACE_OWNERSHIP,
    ) -> None:
        self.layout = layout
        self.slots = slots
        self.specs = tuple(specs)
        self.vocab = list(vocab)
        self.tokenizer_version = tokenizer_version

        if len(self.vocab) != layout.vocab_size:
            raise ValueError(
                f"vocab has {len(self.vocab)} entries but layout allocates "
                f"{layout.vocab_size}"
            )

        self.lookup = SlotLookup(layout)
        self.slot_map: Dict[str, str] = slot_map_from_specs(self.specs)
        structural = [s.name for s in self.specs if s.structural]
        self.structural_slot = structural[0] if structural else CODE_MATH

        # Opt-in in-stream slot marker. This is the ONLY thing a "slot marker id"
        # needs to be: an ordinary special token at a low id, costing one row.
        # There is no magic sentinel. Off unless SLOT_SEP was added to specials.
        self.slot_sep_id: Optional[int] = (
            layout.specials.index(SLOT_SEP) if SLOT_SEP in layout.specials else None
        )
        if emit_slot_markers and self.slot_sep_id is None:
            raise ValueError(
                f"emit_slot_markers requires {SLOT_SEP!r} in special_tokens"
            )
        self.emit_slot_markers = emit_slot_markers and self.slot_sep_id is not None
        if whitespace_ownership not in WHITESPACE_MODES:
            raise ValueError(f"unknown whitespace_ownership {whitespace_ownership!r}")
        self.whitespace_ownership = whitespace_ownership
        self.stats = EncodeStats()

    # --- construction -----------------------------------------------------

    @classmethod
    def from_dir(cls, out_dir: str) -> "PartitionedTokenizer":
        layout = IdLayout.from_json(os.path.join(out_dir, SLOTS_NAME))
        specs = load_specs(out_dir)
        with open(os.path.join(out_dir, VOCAB_NAME), "r", encoding="utf-8") as fh:
            vd = json.load(fh)

        slots: Dict[str, object] = {}
        for rng in layout.slots:
            if rng.algorithm == "bytes":
                slots[rng.name] = ByteFallbackSlot()
            else:
                path = os.path.join(out_dir, MODELS_DIRNAME, f"{rng.name}.json")
                slots[rng.name] = load_hf_slot(rng.name, rng.algorithm, path)

        return cls(
            layout,
            slots,
            vd["tokens"],
            specs,
            vd.get("tokenizer_version", "v1"),
            emit_slot_markers=bool(vd.get("emit_slot_markers", False)),
            whitespace_ownership=vd.get(
                "whitespace_ownership", DEFAULT_WHITESPACE_OWNERSHIP
            ),
        )

    # --- encoding ---------------------------------------------------------

    def _slot_obj(self, name: str):
        obj = self.slots.get(name)
        if obj is None:
            obj = self.slots[CATCHALL]
        return obj

    def _encode_text(
        self, slot_name: str, text: str, out: List[tuple]
    ) -> None:
        """Encode `text` with `slot_name`, degrading per character if needed.

        Fallback granularity matters enormously. Falling back for the whole run
        means one unrepresentable character costs the entire run's tokenisation:
        a stray newline in a 2800-character Devanagari run turned ~2400 tokens
        into 7601, because the whole run went to per-byte CATCHALL. Under
        `leading` runs are whole sentences, so that blast radius is huge.

        Instead, a failing span is bisected until the offending characters are
        isolated, so each one costs only its own bytes and the surrounding text
        keeps its slot's tokenisation. Work is O(k log n) in the number of
        unrepresentable characters k.
        """
        stack: List[tuple] = [(slot_name, text)]
        while stack:
            name, chunk = stack.pop()
            if not chunk:
                continue

            if name == CATCHALL:
                rng = self.layout.slot_by_name(CATCHALL)
                out.append((CATCHALL, self.slots[CATCHALL].encode(chunk)))
                continue

            rng = self.layout.slot_by_name(name)
            try:
                out.append((name, self._slot_obj(name).encode(chunk)))
                continue
            except UnrepresentableRun:
                pass

            if len(chunk) == 1:
                self.stats.fallback_chars += 1
                out.append((CATCHALL, self.slots[CATCHALL].encode(chunk)))
                continue

            mid = len(chunk) // 2
            # push right first so the left half is popped first: order preserved
            stack.append((name, chunk[mid:]))
            stack.append((name, chunk[:mid]))

    def encode_runs(self, text: str, normalize_form: str = "NFC"):
        """Yield (Run, global ids) in original order. Never raises for content.

        The returned ids may span more than one slot internally when a character
        had to fall back; the decoder routes each id by its own range, so this is
        transparent. The yielded Run is the segmentation unit, for reporting.
        """
        for run in segment(
            text,
            normalize_form,
            slot_map=self.slot_map,
            structural_slot=self.structural_slot,
            whitespace_ownership=self.whitespace_ownership,
        ):
            slot_name = run.slot if run.slot in self.slots else CATCHALL
            segments: List[tuple] = []
            self._encode_text(slot_name, run.text, segments)

            ids: List[int] = []
            for seg_slot, local_ids in segments:
                rng = self.layout.slot_by_name(seg_slot)
                self.stats.slot_tokens[seg_slot] += len(local_ids)
                ids.extend(rng.start + l for l in local_ids)
            yield run, ids

    def encode(self, text: str, normalize_form: str = "NFC") -> List[int]:
        ids: List[int] = []
        prev_slot: Optional[str] = None
        for run, run_ids in self.encode_runs(text, normalize_form):
            self.stats.runs += 1
            self.stats.slot_runs[run.slot] += 1
            if (
                self.emit_slot_markers
                and prev_slot is not None
                and run.slot != prev_slot
            ):
                ids.append(self.slot_sep_id)
            ids.extend(run_ids)
            prev_slot = run.slot
        self.stats.ids += len(ids)
        if ids and max(ids) >= self.layout.first_invalid_id:
            raise AssertionError(
                f"emitted id {max(ids)} outside the allocated vocab "
                f"(size {self.layout.vocab_size})"
            )
        return ids

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    # --- decoding ---------------------------------------------------------

    def decode(self, ids: Sequence[int]) -> str:
        parts: List[str] = []
        i = 0
        n = len(ids)
        while i < n:
            tid = ids[i]
            if tid < 0 or tid >= self.layout.vocab_size:
                raise ValueError(
                    f"id {tid} outside allocated vocab "
                    f"(size {self.layout.vocab_size})"
                )
            if tid < self.layout.special_end:
                # Slot markers are structural, not content: skip on decode so
                # that round-trip stays exact.
                if self.slot_sep_id is None or tid != self.slot_sep_id:
                    parts.append(self.vocab[tid])
                i += 1
                continue

            rng = self.lookup.slot_of(tid)
            if rng is None:
                raise AssertionError(f"id {tid} belongs to no slot")
            obj = self.slots[rng.name]

            j = i
            while j < n:
                nxt = ids[j]
                if nxt < self.layout.special_end:
                    break
                nxt_rng = self.lookup.slot_of(nxt)
                if nxt_rng is None or nxt_rng.name != rng.name:
                    break
                j += 1

            local_ids = [v - rng.start for v in ids[i:j]]
            parts.append(obj.decode(local_ids))
            i = j
        return "".join(parts)

    def decode_batch(self, batch: Sequence[Sequence[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch]

    # --- introspection ----------------------------------------------------

    def slot_of_id(self, token_id: int) -> Optional[str]:
        return self.lookup.slot_name(token_id)

    def describe(self) -> dict:
        return {
            "tokenizer_version": self.tokenizer_version,
            "vocab_size": self.layout.vocab_size,
            "first_invalid_id": self.layout.first_invalid_id,
            "padded_vocab_size": self.layout.padded_vocab_size,
            "special_tokens": self.layout.specials,
            "slots": [
                {
                    "name": s.name,
                    "start": s.start,
                    "end": s.end,
                    "size": s.size,
                    "algorithm": s.algorithm,
                    "scripts": list(
                        next(x.scripts for x in self.specs if x.name == s.name)
                    ),
                }
                for s in self.layout.slots
            ],
        }

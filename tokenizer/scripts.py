"""Script detection, normalization, and structural segmentation.

This is the deterministic front end of the tokenizer. Everything here must be
pure, total, and free of machine-learned components: which slot a piece of text
lands in is decided by Unicode properties and by explicit document structure,
never by a classifier. See SPEC.md section 2.

Two layers, deliberately separate:

  * ``script_of`` / ``classify_char`` -- Unicode script detection. Knows nothing
    about slots.
  * ``segment`` with a ``slot_map`` -- routes scripts to slots. The map is data,
    built from the slot specs, so a language is added to a slot by editing config
    and never by editing detection code.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Sequence

# --------------------------------------------------------------------------
# Slot names
# --------------------------------------------------------------------------
# These name *algorithm groups*, never languages. LATIN holds English, Hinglish,
# Kanglish and every other Latin-script language; DEVANAGARI holds Hindi,
# Marathi, Nepali and Sanskrit. Languages sharing a script share a slot.

NEUTRAL = "NEUTRAL"
LATIN = "LATIN"
DEVANAGARI = "DEVANAGARI"
KANNADA = "KANNADA"
CODE_MATH = "CODE_MATH"
CATCHALL = "CATCHALL"

# --------------------------------------------------------------------------
# Script names
# --------------------------------------------------------------------------
# COMMON is Unicode's own term for characters shared by all scripts: whitespace,
# digits, punctuation, symbols, emoji.
COMMON = "COMMON"
# Letters of a script we do not model, plus control characters. Routed to
# CATCHALL, which is lossless by byte fallback.
UNKNOWN = "UNKNOWN"

# Pseudo-script, resolved during segmentation by attaching to the preceding run.
_INHERITED = "INHERITED"

# Unicode block ranges per script. Deliberately a modest explicit table rather
# than a full Scripts.txt: a script absent here is not an error, it simply falls
# through to UNKNOWN and then to CATCHALL, which is lossless. Extending coverage
# means adding ranges here -- no other code changes.
SCRIPT_RANGES: Dict[str, Sequence[tuple[int, int]]] = {
    "LATIN": (
        (0x0041, 0x005A), (0x0061, 0x007A), (0x00AA, 0x00AA), (0x00BA, 0x00BA),
        (0x00C0, 0x00D6), (0x00D8, 0x00F6), (0x00F8, 0x02B8),
        (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF),
    ),
    "GREEK": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "CYRILLIC": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF)),
    "ARMENIAN": ((0x0530, 0x058F),),
    "HEBREW": ((0x0590, 0x05FF),),
    "ARABIC": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "DEVANAGARI": ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF)),
    "BENGALI": ((0x0980, 0x09FF),),
    "GURMUKHI": ((0x0A00, 0x0A7F),),
    "GUJARATI": ((0x0A80, 0x0AFF),),
    "ORIYA": ((0x0B00, 0x0B7F),),
    "TAMIL": ((0x0B80, 0x0BFF),),
    "TELUGU": ((0x0C00, 0x0C7F),),
    "KANNADA": ((0x0C80, 0x0CFF),),
    "MALAYALAM": ((0x0D00, 0x0D7F),),
    "SINHALA": ((0x0D80, 0x0DFF),),
    "THAI": ((0x0E00, 0x0E7F),),
    "LAO": ((0x0E80, 0x0EFF),),
    "TIBETAN": ((0x0F00, 0x0FFF),),
    "MYANMAR": ((0x1000, 0x109F),),
    "GEORGIAN": ((0x10A0, 0x10FF),),
    "ETHIOPIC": ((0x1200, 0x137F),),
    "CHEROKEE": ((0x13A0, 0x13FF),),
    "KHMER": ((0x1780, 0x17FF),),
    "HANGUL": ((0x1100, 0x11FF), (0xAC00, 0xD7AF), (0x3130, 0x318F)),
    "HIRAGANA": ((0x3040, 0x309F),),
    "KATAKANA": ((0x30A0, 0x30FF),),
    "CJK": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)),
}

# Flattened + sorted for binary search.
_BLOCKS: List[tuple[int, int, str]] = sorted(
    ((lo, hi, name) for name, ranges in SCRIPT_RANGES.items() for lo, hi in ranges),
    key=lambda b: b[0],
)
_BLOCK_LO: List[int] = [b[0] for b in _BLOCKS]

# --------------------------------------------------------------------------
# Default script -> slot routing
# --------------------------------------------------------------------------
# DATA, not logic. slots.py derives its specs from this and asserts they agree,
# so the two can never drift.

DEFAULT_SLOT_MAP: Dict[str, str] = {
    COMMON: NEUTRAL,
    LATIN: LATIN,
    DEVANAGARI: DEVANAGARI,
    KANNADA: KANNADA,
}

_WHITESPACE_CONTROLS = frozenset("\t\n\v\f\r")
_JOINERS = frozenset({0x200C, 0x200D})
_VARIATION_SELECTORS = (0xFE00, 0xFE0F)

# Structural spans that override script classification. Order is priority: the
# first pattern that matches at a position wins. Fenced blocks before inline
# spans so that a fence is not eaten by an inner backtick.
#
# Every delimiter pair must CLOSE. An earlier version allowed `|$` (match to end of
# input) so truncated code blocks still registered as CODE_MATH. That was a serious
# bug: on a large concatenated corpus, one unterminated ``` or $$ swallowed the
# remainder of the input, routing megabytes of Devanagari prose into a slot whose
# alphabet had been seeded from code bytes only. That slot then raised
# UnrepresentableRun and fell back to per-byte encoding, inflating the token count
# ~4.5x and producing a confident, wrong bits-per-byte verdict. An unterminated
# delimiter is now simply not structural, which is both safe and predictable.
_STRUCT_PATTERNS: Sequence[re.Pattern[str]] = tuple(
    re.compile(p)
    for p in (
        r"```[\s\S]*?```",                # fenced code (must close)
        r"\$\$[\s\S]*?\$\$",              # display math (must close)
        r"\\\[[\s\S]*?\\\]",              # \[ ... \]
        r"\\\([\s\S]*?\\\)",              # \( ... \)
        r"\$[^$\n]{1,200}?\$",            # inline math
        r"`[^`\n]{1,200}?`",              # inline code
    )
)


@dataclass(frozen=True)
class Run:
    """A maximal span of one class, in original text order."""

    slot: str
    text: str
    start: int  # char offset in the normalized text
    end: int  # exclusive
    script: str = COMMON

    def __len__(self) -> int:
        return self.end - self.start


def normalize(text: str, form: str = "NFC") -> str:
    """Normalize text. NFC by default.

    NFKC is deliberately not used: it is not invertible and folds Indic
    letterforms in ways that are linguistically wrong for Devanagari and
    Kannada. ZWJ/ZWNJ survive NFC untouched.
    """
    if form == "none":
        return text
    return unicodedata.normalize(form, text)


def _script_of_slow(ch: str) -> str:
    cat = unicodedata.category(ch)
    head = cat[0]

    # Whitespace first: \t and \n are Cc, not Z*.
    if head == "Z" or ch in _WHITESPACE_CONTROLS:
        return COMMON
    # Digits, punctuation, symbols (emoji, arrows, math operators, currency).
    if head in ("N", "P", "S"):
        return COMMON

    cp = ord(ch)
    i = bisect.bisect_right(_BLOCK_LO, cp) - 1
    if i >= 0:
        lo, hi, name = _BLOCKS[i]
        if lo <= cp <= hi:
            return name

    # Marks outside a modelled block (e.g. combining accents U+0300..) attach to
    # whatever preceded them.
    if cat in ("Mn", "Mc", "Me"):
        return _INHERITED
    if cp in _JOINERS or _VARIATION_SELECTORS[0] <= cp <= _VARIATION_SELECTORS[1]:
        return _INHERITED

    return UNKNOWN


# Precomputed direct lookup for the Unicode Basic Multilingual Plane (0x0000..0xFFFF).
# Accelerates character classification by 4.5x-5x in pure Python.
_BMP_SCRIPT_TABLE: tuple[str, ...] = tuple(_script_of_slow(chr(i)) for i in range(65536))


def script_of(ch: str) -> str:
    """Unicode script of a single character.

    Returns a script name, COMMON, UNKNOWN, or _INHERITED. Knows nothing about
    slots.
    """
    cp = ord(ch)
    if cp < 65536:
        return _BMP_SCRIPT_TABLE[cp]
    return _script_of_slow(ch)


def classify_char(ch: str, slot_map: Mapping[str, str] | None = None) -> str:
    """Slot name for a single character, under `slot_map`.

    Convenience wrapper for callers that want the slot directly. Prefer
    ``script_of`` when the script itself matters.
    """
    smap = DEFAULT_SLOT_MAP if slot_map is None else slot_map
    script = script_of(ch)
    if script == _INHERITED:
        return _INHERITED
    return smap.get(script, CATCHALL)


def structural_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield (start, end) spans of code/math detected by document structure.

    Left-to-right greedy scan: at each position the highest-priority pattern that
    matches wins, so overlapping candidates cannot both be taken.
    """
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos] not in "`$\\":
            pos += 1
            continue
        for pat in _STRUCT_PATTERNS:
            m = pat.match(text, pos)
            if m is not None and m.end() > m.start():
                yield (pos, m.end())
                pos = m.end()
                break
        else:
            pos += 1


# How whitespace is owned (SPEC.md 5.1):
#   "neutral" -- whitespace is its own NEUTRAL run. Unambiguous, but every space
#                costs a token and no script slot can learn a piece spanning a
#                space.
#   "leading" -- whitespace joins the run that follows it (SentencePiece's U+2581
#                convention), so " word" is a single piece opportunity. Trailing
#                whitespace joins the preceding run.
WHITESPACE_NEUTRAL = "neutral"
WHITESPACE_LEADING = "leading"
WHITESPACE_MODES = (WHITESPACE_NEUTRAL, WHITESPACE_LEADING)

# The product default. Measurement decided this, not preference: on a
# sequence-novel natural held-out set, `neutral` lost to the shared-vocab
# baseline on Devanagari (-62%) and Latin (-66%), while `leading` beat it on
# every script (Devanagari +24%, Kannada +48%, Latin +18%). `neutral` costs a
# token per space and forbids cross-word merges. See SPEC.md 5.1 and 12.
DEFAULT_WHITESPACE_OWNERSHIP = WHITESPACE_LEADING


def _is_whitespace(ch: str) -> bool:
    return ch in _WHITESPACE_CONTROLS or unicodedata.category(ch)[0] == "Z"


def _classify_all(
    text: str,
    slot_map: Mapping[str, str],
    structural_slot: str,
    whitespace_ownership: str = WHITESPACE_NEUTRAL,
) -> tuple[List[str], List[str]]:
    """Per-character (slot, script) with structural override applied."""
    if whitespace_ownership not in WHITESPACE_MODES:
        raise ValueError(
            f"unknown whitespace_ownership {whitespace_ownership!r}; "
            f"expected one of {WHITESPACE_MODES}"
        )

    n = len(text)
    code_mask = bytearray(n)
    for s, e in structural_spans(text):
        for i in range(s, e):
            code_mask[i] = 1

    slots: List[str] = [NEUTRAL] * n
    scripts: List[str] = [COMMON] * n
    is_ws = bytearray(n)
    prev_slot = NEUTRAL
    prev_script = COMMON

    for i, ch in enumerate(text):
        if code_mask[i]:
            slots[i] = structural_slot
            scripts[i] = CODE_MATH
            continue
        script = script_of(ch)
        if script == _INHERITED:
            slots[i] = prev_slot
            scripts[i] = prev_script
            continue
        slots[i] = slot_map.get(script, CATCHALL)
        scripts[i] = script
        prev_slot = slots[i]
        prev_script = script
        # Whitespace inside a structural span belongs to that span and is left
        # alone; only inter-run whitespace is a candidate for reassignment.
        if _is_whitespace(ch):
            is_ws[i] = 1

    if whitespace_ownership == WHITESPACE_LEADING:
        _attach_whitespace_forward(slots, scripts, is_ws)
    return slots, scripts


def _attach_whitespace_forward(
    slots: List[str], scripts: List[str], is_ws: bytearray
) -> None:
    """Move each inter-run whitespace run into the run that follows it.

    In place. A whitespace run with nothing after it joins what precedes it; text
    that is entirely whitespace stays NEUTRAL. Characters are only *reassigned*
    between runs, never reordered or dropped, so concatenating the runs still
    reproduces the input exactly.
    """
    n = len(slots)
    i = 0
    while i < n:
        if not is_ws[i]:
            i += 1
            continue
        j = i
        while j < n and is_ws[j]:
            j += 1

        if j < n:
            target_slot, target_script = slots[j], scripts[j]
        else:
            k = i - 1
            while k >= 0 and is_ws[k]:
                k -= 1
            if k >= 0:
                target_slot, target_script = slots[k], scripts[k]
            else:
                target_slot, target_script = NEUTRAL, COMMON

        for t in range(i, j):
            slots[t] = target_slot
            scripts[t] = target_script
        i = j


def segment(
    text: str,
    normalize_form: str = "NFC",
    slot_map: Mapping[str, str] | None = None,
    structural_slot: str = CODE_MATH,
    whitespace_ownership: str = DEFAULT_WHITESPACE_OWNERSHIP,
) -> List[Run]:
    """Split text into ordered Runs, one per contiguous same-slot span.

    Order is preserved: the i-th run's tokens occupy a contiguous id span in the
    encoded sequence, and runs appear in input order (SPEC.md section 7).
    """
    text = normalize(text, normalize_form)
    if not text:
        return []

    slots, scripts = _classify_all(
        text,
        DEFAULT_SLOT_MAP if slot_map is None else slot_map,
        structural_slot,
        whitespace_ownership,
    )
    runs: List[Run] = []
    start = 0
    cur = slots[0]
    for i in range(1, len(text)):
        if slots[i] != cur:
            runs.append(Run(cur, text[start:i], start, i, scripts[start]))
            start = i
            cur = slots[i]
    runs.append(Run(cur, text[start:], start, len(text), scripts[start]))
    return runs

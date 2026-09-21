"""Script-partitioned multilingual tokenizer.

The id space is partitioned into contiguous slots. A slot is declared as
``(name, algorithm, scripts[])``, so languages sharing a script and an algorithm
share a slot, and different slots may run different algorithms. Slot boundaries
are metadata recovered by lookup, so nothing is emitted into the token stream to
mark language. See SPEC.md.
"""

from .scripts import (
    CATCHALL,
    CODE_MATH,
    COMMON,
    DEFAULT_WHITESPACE_OWNERSHIP,
    DEVANAGARI,
    KANNADA,
    LATIN,
    NEUTRAL,
    UNKNOWN,
    WHITESPACE_LEADING,
    WHITESPACE_MODES,
    WHITESPACE_NEUTRAL,
    Run,
    normalize,
    script_of,
    segment,
)
from .slots import (
    BYTE_TABLE_SIZE,
    DEFAULT_EMBED_MULTIPLE,
    DEFAULT_SPECS,
    SLOT_SEP,
    IdLayout,
    SlotLookup,
    SlotRange,
    SlotSpec,
    slot_map_from_specs,
    specs_from_dicts,
)

from .partitioned import PartitionedTokenizer
from .hf_tokenizer import IndicPartitionedTokenizer

__all__ = [
    "PartitionedTokenizer",
    "IndicPartitionedTokenizer",

    "BYTE_TABLE_SIZE",
    "CATCHALL",
    "CODE_MATH",
    "COMMON",
    "DEFAULT_EMBED_MULTIPLE",
    "DEFAULT_SPECS",
    "DEFAULT_WHITESPACE_OWNERSHIP",
    "DEVANAGARI",
    "KANNADA",
    "LATIN",
    "NEUTRAL",
    "SLOT_SEP",
    "UNKNOWN",
    "WHITESPACE_LEADING",
    "WHITESPACE_MODES",
    "WHITESPACE_NEUTRAL",
    "IdLayout",
    "Run",
    "SlotLookup",
    "SlotRange",
    "SlotSpec",
    "normalize",
    "script_of",
    "segment",
    "slot_map_from_specs",
    "specs_from_dicts",
]

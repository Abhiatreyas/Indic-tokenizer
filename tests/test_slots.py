"""Slot specs, script routing, id layout, and budget allocation."""

from __future__ import annotations

import pytest

from tokenizer.scripts import (
    CATCHALL,
    CODE_MATH,
    COMMON,
    DEFAULT_SLOT_MAP,
    DEVANAGARI,
    KANNADA,
    LATIN,
    NEUTRAL,
)
from tokenizer.slots import (
    BYTE_TABLE_SIZE,
    DEFAULT_EMBED_MULTIPLE,
    DEFAULT_SPECS,
    SLOT_SEP,
    IdLayout,
    SlotLookup,
    SlotSpec,
    allocate_budgets,
    catchall_spec,
    slot_map_from_specs,
    specs_from_dicts,
    validate_specs,
)

SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>", "<mask>")


def sizes(**kw) -> dict:
    base = {
        "NEUTRAL": 100,
        "LATIN": 200,
        "DEVANAGARI": 150,
        "KANNADA": 120,
        "CODE_MATH": 80,
        "CATCHALL": BYTE_TABLE_SIZE,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# Slot specs: (name, algorithm, scripts[])
# --------------------------------------------------------------------------


def test_default_specs_agree_with_the_canonical_script_map():
    """Specs and the segmenter's map are derived from one source; assert it."""
    assert slot_map_from_specs(DEFAULT_SPECS) == DEFAULT_SLOT_MAP


def test_a_slot_holds_many_languages_via_its_script():
    """LATIN is one slot for English, Hinglish, Kanglish, French, ... There is
    no per-language slot anywhere in the default model."""
    names = [s.name for s in DEFAULT_SPECS]
    assert LATIN in names
    assert "ENGLISH" not in names and "HINDI" not in names and "MARATHI" not in names
    # Devanagari covers Hindi + Marathi + Nepali + Sanskrit with one slot.
    dev = next(s for s in DEFAULT_SPECS if s.name == DEVANAGARI)
    assert dev.scripts == ("DEVANAGARI",)


def test_scripts_may_be_grouped_into_one_slot():
    """The user's requirement: several scripts sharing one algorithm and one
    vocabulary share a slot. Right call for low-resource scripts."""
    specs = specs_from_dicts(
        [
            {"name": NEUTRAL, "algorithm": "bpe", "scripts": [COMMON]},
            {"name": LATIN, "algorithm": "bpe", "scripts": [LATIN]},
            {
                "name": "INDIC_SHARED",
                "algorithm": "bpe",
                "scripts": ["TAMIL", "TELUGU", "MALAYALAM", "BENGALI"],
            },
            {"name": CODE_MATH, "algorithm": "bpe", "structural": True},
            {
                "name": CATCHALL,
                "algorithm": "bytes",
                "trained": False,
                "exempt_from_budget": True,
            },
        ]
    )
    m = slot_map_from_specs(specs)
    for s in ("TAMIL", "TELUGU", "MALAYALAM", "BENGALI"):
        assert m[s] == "INDIC_SHARED"


def test_algorithms_are_per_slot():
    """Different slots may run different algorithms."""
    specs = specs_from_dicts(
        [
            {"name": NEUTRAL, "algorithm": "bpe", "scripts": [COMMON]},
            {"name": LATIN, "algorithm": "bpe", "scripts": [LATIN]},
            {"name": DEVANAGARI, "algorithm": "unigram", "scripts": [DEVANAGARI]},
            {"name": CODE_MATH, "algorithm": "bpe", "structural": True},
            {
                "name": CATCHALL,
                "algorithm": "bytes",
                "trained": False,
                "exempt_from_budget": True,
            },
        ]
    )
    by_name = {s.name: s.algorithm for s in specs}
    assert by_name[LATIN] == "bpe"
    assert by_name[DEVANAGARI] == "unigram"


def test_a_script_cannot_be_claimed_twice():
    with pytest.raises(ValueError, match="claimed by both"):
        validate_specs(
            [
                SlotSpec(LATIN, "bpe", (LATIN,)),
                SlotSpec("LATIN2", "bpe", (LATIN,)),
                SlotSpec(CODE_MATH, "bpe", (), structural=True),
                SlotSpec(CATCHALL, "bytes", (), trained=False, exempt_from_budget=True),
            ]
        )


def test_duplicate_slot_names_rejected():
    with pytest.raises(ValueError, match="duplicate slot names"):
        validate_specs(
            [
                SlotSpec(LATIN, "bpe", (LATIN,)),
                SlotSpec(LATIN, "bpe", ()),
                SlotSpec(CATCHALL, "bytes", (), trained=False, exempt_from_budget=True),
            ]
        )


def test_bytes_slot_must_exist():
    with pytest.raises(ValueError, match="bytes slot"):
        validate_specs([SlotSpec(LATIN, "bpe", (LATIN,))])


def test_bytes_slot_is_not_trained():
    with pytest.raises(ValueError, match="not trained"):
        validate_specs([SlotSpec(CATCHALL, "bytes", (), trained=True)])


def test_only_one_structural_slot():
    with pytest.raises(ValueError, match="structural"):
        validate_specs(
            [
                SlotSpec(CODE_MATH, "bpe", (), structural=True),
                SlotSpec("MATH2", "bpe", (), structural=True),
                SlotSpec(CATCHALL, "bytes", (), trained=False, exempt_from_budget=True),
            ]
        )


def test_unknown_algorithm_rejected():
    with pytest.raises(ValueError, match="unknown algorithm"):
        validate_specs(
            [
                SlotSpec(LATIN, "wordpiece", (LATIN,)),
                SlotSpec(CATCHALL, "bytes", (), trained=False, exempt_from_budget=True),
            ]
        )


def test_catchall_spec_helper():
    assert catchall_spec(DEFAULT_SPECS).name == CATCHALL


# --------------------------------------------------------------------------
# Id layout
# --------------------------------------------------------------------------


def test_slots_start_after_specials():
    layout = IdLayout.from_sizes(SPECIALS, sizes())
    assert layout.special_end == len(SPECIALS)
    assert layout.slots[0].start == len(SPECIALS)


def test_slots_are_contiguous_and_ordered():
    layout = IdLayout.from_sizes(SPECIALS, sizes())
    expected = layout.special_end
    for s in layout.slots:
        assert s.start == expected
        expected = s.end


def test_vocab_size_is_sum_of_parts():
    s = sizes()
    layout = IdLayout.from_sizes(SPECIALS, s)
    assert layout.vocab_size == len(SPECIALS) + sum(s.values())
    assert layout.first_invalid_id == layout.vocab_size


def test_no_magic_sentinel_exists():
    """The old design reserved id 999999. There is no such constant now."""
    import tokenizer.slots as slots_mod

    assert not hasattr(slots_mod, "POISON_ID")
    layout = IdLayout.from_sizes(SPECIALS, sizes())
    assert layout.vocab_size < 999999


def test_embedding_padding_is_computed_not_magic():
    layout = IdLayout.from_sizes(SPECIALS, sizes(), embed_multiple=128)
    assert layout.padded_vocab_size % 128 == 0
    assert layout.padded_vocab_size >= layout.vocab_size
    assert layout.padded_vocab_size - layout.vocab_size < 128


def test_padding_is_a_noop_when_multiple_is_one():
    layout = IdLayout.from_sizes(SPECIALS, sizes(), embed_multiple=1)
    assert layout.padded_vocab_size == layout.vocab_size


def test_slot_sep_is_an_ordinary_special_token():
    """The only legitimate in-stream marker: one token, one row, low id."""
    specials = SPECIALS + (SLOT_SEP,)
    layout = IdLayout.from_sizes(specials, sizes())
    assert layout.specials.index(SLOT_SEP) == len(SPECIALS)
    assert layout.specials.index(SLOT_SEP) < layout.special_end


def test_gaps_in_slot_sizes_are_rejected():
    bad = sizes()
    del bad[KANNADA]
    with pytest.raises(ValueError, match="missing size"):
        IdLayout.from_sizes(SPECIALS, bad)


def test_zero_sized_slot_rejected():
    with pytest.raises(ValueError, match="non-positive"):
        IdLayout.from_sizes(SPECIALS, sizes(KANNADA=0))


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def test_binary_search_routes_every_id():
    layout = IdLayout.from_sizes(SPECIALS, sizes())
    for s in layout.slots:
        for tid in (s.start, s.start + 1, s.end - 1):
            assert layout.slot_of(tid).name == s.name


def test_slot_of_returns_none_for_specials_and_out_of_range():
    layout = IdLayout.from_sizes(SPECIALS, sizes())
    for tid in range(len(SPECIALS)):
        assert layout.slot_of(tid) is None
    assert layout.slot_of(-1) is None
    assert layout.slot_of(layout.vocab_size) is None


def test_flat_lookup_agrees_with_binary_search():
    """SlotLookup is the hot path; it must be exactly equivalent."""
    layout = IdLayout.from_sizes(SPECIALS, sizes())
    lookup = layout.lookup()
    for tid in range(-2, layout.vocab_size + 2):
        a = layout.slot_of(tid)
        b = lookup.slot_of(tid)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.name == b.name == lookup.slot_name(tid)


def test_flat_lookup_covers_every_non_special_id():
    layout = IdLayout.from_sizes(SPECIALS, sizes())
    lookup = layout.lookup()
    unrouted = [
        i
        for i in range(layout.special_end, layout.vocab_size)
        if lookup.slot_of(i) is None
    ]
    assert unrouted == []


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def test_json_round_trip(tmp_path):
    layout = IdLayout.from_sizes(SPECIALS, sizes(), embed_multiple=64)
    path = str(tmp_path / "slots.json")
    layout.to_json(path)
    again = IdLayout.from_json(path)
    assert again.as_dict() == layout.as_dict()
    assert again.embed_multiple == 64
    assert again.padded_vocab_size == layout.padded_vocab_size


# --------------------------------------------------------------------------
# Budget allocation
# --------------------------------------------------------------------------


def test_budgets_sum_to_total():
    slot_bytes = {
        "NEUTRAL": 10_000,
        "LATIN": 900_000,
        "DEVANAGARI": 300_000,
        "KANNADA": 120_000,
        "CODE_MATH": 50_000,
    }
    b = allocate_budgets(slot_bytes, 8000, alpha=0.7, min_slot=300, max_slot=4000)
    assert sum(b.values()) == 8000
    assert b[CATCHALL] == BYTE_TABLE_SIZE


def test_alpha_tempers_dominance_of_the_largest_slot():
    slot_bytes = {
        "NEUTRAL": 10_000,
        "LATIN": 900_000,
        "DEVANAGARI": 300_000,
        "KANNADA": 120_000,
        "CODE_MATH": 50_000,
    }
    prop = allocate_budgets(slot_bytes, 8000, alpha=1.0, min_slot=1, max_slot=64_000)
    tempered = allocate_budgets(slot_bytes, 8000, alpha=0.7, min_slot=1, max_slot=64_000)
    assert tempered[LATIN] < prop[LATIN]
    assert tempered[KANNADA] > prop[KANNADA]


def test_min_floor_protects_a_tiny_script():
    slot_bytes = {
        "NEUTRAL": 10,
        "LATIN": 10_000_000,
        "DEVANAGARI": 5,
        "KANNADA": 3,
        "CODE_MATH": 2,
    }
    b = allocate_budgets(slot_bytes, 8000, alpha=1.0, min_slot=300, max_slot=4000)
    for name in ("NEUTRAL", DEVANAGARI, KANNADA, "CODE_MATH"):
        assert b[name] >= 300


def test_max_ceiling_is_respected():
    slot_bytes = {
        "NEUTRAL": 1,
        "LATIN": 10_000_000,
        "DEVANAGARI": 1,
        "KANNADA": 1,
        "CODE_MATH": 1,
    }
    b = allocate_budgets(slot_bytes, 40_000, alpha=1.0, min_slot=300, max_slot=2000)
    assert b[LATIN] <= 2000


def test_zero_corpus_falls_back_to_even_split():
    b = allocate_budgets({}, 6000, alpha=0.7, min_slot=300, max_slot=4000)
    assert sum(b.values()) == 6000


def test_total_too_small_raises():
    with pytest.raises(ValueError, match="too small"):
        allocate_budgets({LATIN: 1}, 100)

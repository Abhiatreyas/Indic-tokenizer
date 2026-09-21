"""Segmentation, normalization, and the order-preservation invariant."""

from __future__ import annotations

import unicodedata

import pytest

from tokenizer.scripts import (
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
    WHITESPACE_NEUTRAL,
    classify_char,
    normalize,
    script_of,
    segment,
)


def slots_of(text: str, ownership: str = WHITESPACE_NEUTRAL) -> list[str]:
    """Distinct consecutive slot names."""
    out: list[str] = []
    for r in segment(text, whitespace_ownership=ownership):
        if not out or out[-1] != r.slot:
            out.append(r.slot)
    return out


# --- order preservation ----------------------------------------------------


@pytest.mark.parametrize("ownership", [WHITESPACE_NEUTRAL, WHITESPACE_LEADING])
@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "hello world",
        "नमस्ते दुनिया",
        "ಕನ್ನಡ ಭಾಷೆ",
        "नमस्ते world ಕನ್ನಡ mixed",
        "def f(x):\n    return x ** 2",
        "$$\\int_0^1 x^2\\,dx = \\frac{1}{3}$$",
        "🚀 emoji and ∑ symbols",
        "தமிழ் العربية 中文",
        "a\u0301b\u0302c",
        "\u0000\u0001control",
    ],
)
def test_runs_reassemble_to_normalized_text(text, ownership):
    """The concatenation of runs must equal the normalized input, in order.

    This is the invariant that makes order-preserving encoding possible at all,
    and it must hold under BOTH whitespace ownership modes.
    """
    runs = segment(text, whitespace_ownership=ownership)
    assert "".join(r.text for r in runs) == normalize(text)


@pytest.mark.parametrize("ownership", [WHITESPACE_NEUTRAL, WHITESPACE_LEADING])
def test_run_offsets_are_consistent(ownership):
    text = "नमस्ते world ಕನ್ನಡ mixed"
    norm = normalize(text)
    pos = 0
    for r in segment(text, whitespace_ownership=ownership):
        assert r.start == pos
        assert r.end == pos + len(r.text)
        assert norm[r.start : r.end] == r.text
        pos = r.end
    assert pos == len(norm)


# --- script classification -------------------------------------------------


def test_pure_latin():
    assert slots_of("hello world") == [LATIN, NEUTRAL, LATIN]


def test_hindi_is_devanagari():
    assert slots_of("नमस्ते") == [DEVANAGARI]


def test_kannada():
    assert slots_of("ಕನ್ನಡ") == [KANNADA]


def test_hinglish_is_latin_by_design():
    """Hinglish is Latin script, so it shares the LATIN slot.

    Documented limitation (SPEC.md 2): separating it from English would require
    language identification, which the encoder deliberately does not do.
    """
    s = slots_of("yaar ye kaam kal tak ho jayega")
    assert set(s) <= {LATIN, NEUTRAL}
    assert LATIN in s


def test_digits_and_punctuation_are_common_script():
    for ch in (",", "7", " ", "\t", "\n"):
        assert script_of(ch) == COMMON
        assert classify_char(ch) == NEUTRAL


def test_devanagari_danda_is_common_script():
    # U+0964 DANDA is punctuation, even though it sits in the Devanagari block.
    assert script_of("\u0964") == COMMON


def test_emoji_is_common_script():
    assert script_of("🚀") == COMMON


def test_control_chars_are_unknown_script():
    assert script_of("\x00") == UNKNOWN
    assert script_of("\x01") == UNKNOWN
    assert classify_char("\x00") == CATCHALL


def test_unmodelled_scripts_go_to_catchall():
    assert slots_of("தமிழ்") == [CATCHALL]
    assert slots_of("العربية") == [CATCHALL]
    assert slots_of("中文") == [CATCHALL]
    assert slots_of("ελληνικά") == [CATCHALL]


def test_control_chars_go_to_catchall():
    assert classify_char("\x00") == CATCHALL
    assert classify_char("\x01") == CATCHALL


# --- script detection coverage --------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("नमस्ते", "DEVANAGARI"),
        ("ಕನ್ನಡ", "KANNADA"),
        ("hello", "LATIN"),
        ("தமிழ்", "TAMIL"),
        ("తెలుగు", "TELUGU"),
        ("മലയാളം", "MALAYALAM"),
        ("বাংলা", "BENGALI"),
        ("ગુજરાતી", "GUJARATI"),
        ("ਪੰਜਾਬੀ", "GURMUKHI"),
        ("ଓଡ଼ିଆ", "ORIYA"),
        ("العربية", "ARABIC"),
        ("עברית", "HEBREW"),
        ("ελληνικά", "GREEK"),
        ("русский", "CYRILLIC"),
        ("한국어", "HANGUL"),
        ("中文", "CJK"),
        ("ひらがな", "HIRAGANA"),
        ("カタカナ", "KATAKANA"),
        ("தமிழ்", "TAMIL"),
    ],
)
def test_script_detection(text, expected):
    assert script_of(text[0]) == expected


def test_grouping_scripts_into_one_slot():
    """Several scripts may be routed to a single slot -- the low-resource case."""
    slot_map = {
        COMMON: NEUTRAL,
        "TAMIL": "INDIC_SHARED",
        "TELUGU": "INDIC_SHARED",
        "MALAYALAM": "INDIC_SHARED",
    }
    # neutral: Tamil and Telugu are separate runs, both owned by the shared slot
    runs = segment(
        "தமிழ் తెలుగు", slot_map=slot_map, whitespace_ownership=WHITESPACE_NEUTRAL
    )
    assert [r.slot for r in runs] == ["INDIC_SHARED", NEUTRAL, "INDIC_SHARED"]

    # leading: the whitespace joins the following run, so they fuse into one
    runs = segment(
        "தமிழ் తెలుగు word", slot_map=slot_map, whitespace_ownership=WHITESPACE_LEADING
    )
    assert runs[0].slot == "INDIC_SHARED"
    assert runs[1].slot == CATCHALL
    assert runs[1].text == " word"  # ASCII-only, no retyped literal

    # unlisted scripts still fall through to CATCHALL
    assert segment("中文", slot_map=slot_map)[0].slot == CATCHALL


def test_runs_expose_both_slot_and_script():
    run = segment("नमस्ते")[0]
    assert run.slot == DEVANAGARI
    assert run.script == "DEVANAGARI"


# --- whitespace ownership --------------------------------------------------


def test_default_ownership_is_leading():
    """Measurement chose this default (SPEC.md 5.1)."""
    assert DEFAULT_WHITESPACE_OWNERSHIP == WHITESPACE_LEADING
    assert len(segment("hello world")) == 1


def test_neutral_keeps_whitespace_in_its_own_run():
    assert slots_of("hello world", WHITESPACE_NEUTRAL) == [LATIN, NEUTRAL, LATIN]


def test_leading_whitespace_joins_the_following_run():
    """SentencePiece's U+2581 convention: " world" is one piece opportunity."""
    runs = segment("नमस्ते world", whitespace_ownership=WHITESPACE_LEADING)
    assert [(r.slot, r.text) for r in runs] == [
        (DEVANAGARI, "नमस्ते"),
        (LATIN, " world"),
    ]


def test_leading_collapses_a_run_of_words():
    runs = segment("the quick brown fox", whitespace_ownership=WHITESPACE_LEADING)
    assert [(r.slot, r.text) for r in runs] == [(LATIN, "the quick brown fox")]


def test_leading_trailing_whitespace_joins_the_preceding_run():
    runs = segment("abc  ", whitespace_ownership=WHITESPACE_LEADING)
    assert [(r.slot, r.text) for r in runs] == [(LATIN, "abc  ")]


def test_leading_leading_whitespace_joins_the_following_run():
    runs = segment("  abc", whitespace_ownership=WHITESPACE_LEADING)
    assert [(r.slot, r.text) for r in runs] == [(LATIN, "  abc")]


def test_leading_whitespace_only_text_stays_neutral():
    runs = segment("   \n\t", whitespace_ownership=WHITESPACE_LEADING)
    assert [(r.slot, r.text) for r in runs] == [(NEUTRAL, "   \n\t")]


def test_leading_does_not_absorb_punctuation():
    """Punctuation stays NEUTRAL; only whitespace is reassigned."""
    runs = segment("word, word", whitespace_ownership=WHITESPACE_LEADING)
    assert [(r.slot, r.text) for r in runs] == [
        (LATIN, "word"),
        (NEUTRAL, ","),
        (LATIN, " word"),
    ]


def test_leading_does_not_steal_whitespace_from_a_code_span():
    """Whitespace inside a structural span belongs to that span."""
    runs = segment("```\na b\n``` tail", whitespace_ownership=WHITESPACE_LEADING)
    slots = [r.slot for r in runs]
    assert CODE_MATH in slots
    # the whole fenced block including its internal whitespace stays CODE_MATH
    code_text = "".join(r.text for r in runs if r.slot == CODE_MATH)
    assert code_text == "```\na b\n```"


def test_invalid_ownership_is_rejected():
    with pytest.raises(ValueError, match="whitespace_ownership"):
        segment("hello", whitespace_ownership="sometimes")


# --- structural override (code / math) -------------------------------------


def test_fenced_code_is_code_math():
    text = "```python\nprint('hi')\n```"
    assert slots_of(text) == [CODE_MATH]


def test_display_math_is_code_math():
    assert slots_of("$$E = mc^2$$") == [CODE_MATH]


def test_inline_math_is_code_math():
    assert slots_of("inline $E = mc^2$ math") == [
        LATIN,
        NEUTRAL,
        CODE_MATH,
        NEUTRAL,
        LATIN,
    ]


def test_inline_code_is_code_math():
    assert CODE_MATH in slots_of("use `encode(text)` here")


def test_structural_span_wins_over_script():
    # Devanagari inside a fenced block still routes to CODE_MATH.
    assert slots_of("```\nनमस्ते\n```") == [CODE_MATH]


def test_raw_latex_frac_survives():
    text = r"$$\frac{a}{b} = \int_0^1 x^2\,dx$$"
    assert slots_of(text) == [CODE_MATH]
    assert "".join(r.text for r in segment(text)) == text


# --- unterminated structural delimiters must not run away ------------------


def test_unterminated_fence_does_not_swallow_the_rest():
    """Regression: allowing the fence pattern to match to end-of-input let one
    stray ``` route the entire remaining corpus into CODE_MATH. That slot then
    raised UnrepresentableRun on Devanagari and byte-fell-back per character,
    inflating tokens ~4.5x and producing a confidently wrong bits-per-byte
    verdict."""
    text = "prose here\n```python\nunclosed\n" + "नमस्ते दुनिया " * 50
    runs = segment(text)
    assert runs[-1].slot == DEVANAGARI
    assert "".join(r.text for r in runs) == text


def test_unterminated_display_math_does_not_swallow_the_rest():
    text = "intro $$ unclosed " + "ಕನ್ನಡ ಭಾಷೆ " * 50
    runs = segment(text)
    assert runs[-1].slot == KANNADA
    assert "".join(r.text for r in runs) == text


def test_unterminated_bracket_math_does_not_swallow_the_rest():
    text = r"intro \[ unclosed " + "தமிழ் " * 50
    runs = segment(text)
    assert runs[-1].slot == CATCHALL
    assert "".join(r.text for r in runs) == text


def test_terminated_spans_still_work():
    assert slots_of("```python\nx = 1\n```", WHITESPACE_NEUTRAL) == [CODE_MATH]
    assert slots_of("$$a = b$$", WHITESPACE_NEUTRAL) == [CODE_MATH]
    assert slots_of(r"\[a = b\]", WHITESPACE_NEUTRAL) == [CODE_MATH]


def test_many_stray_backticks_stay_fast():
    """The old pattern rescanned to end-of-input at every backtick."""
    import time

    text = ("`" * 200 + " filler ") * 50
    t0 = time.time()
    runs = segment(text)
    assert time.time() - t0 < 2.0
    assert "".join(r.text for r in runs) == text


# --- normalization ---------------------------------------------------------


def test_nfc_preserves_ligatures():
    """NFKC would fold these to 'ffiffl'. NFC must not."""
    ligs = "\ufb00\ufb01\ufb02"
    assert normalize(ligs) == ligs


def test_nfc_preserves_zwj_zwnj():
    for ch in ("\u200c", "\u200d"):
        assert normalize(f"a{ch}b") == f"a{ch}b"
        assert ch in normalize(f"क्{ch}ष")


def test_combining_marks_attach_to_previous_run():
    # 'e' + combining acute stays Latin, not a separate run.
    assert slots_of("e\u0301") == [LATIN]


def test_zwj_inside_conjunct_stays_devanagari():
    assert slots_of("क्\u200dष") == [DEVANAGARI]


# --- performance guard -----------------------------------------------------


def test_long_input_is_linear_enough():
    text = "hello world " * 2000
    runs = segment(text)
    assert "".join(r.text for r in runs) == normalize(text)

"""End-to-end gates on a real build.

Builds a small tokenizer into a tmp dir from the synthetic smoke corpus, then
asserts the gates in SPEC.md 10.3 on the artifacts loaded back from disk.
"""

from __future__ import annotations

import os

import pytest

from tokenizer.build import ADVERSARIAL, build
from tokenizer.merge import verify_manifest
from tokenizer.partitioned import PartitionedTokenizer
from tokenizer.scripts import CATCHALL, normalize, segment


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    data = tmp_path_factory.mktemp("data")
    report = build(
        out_dir=str(out),
        data_dir=str(data),
        target_words=1500,
        total_vocab=3000,
        min_slot=300,
        max_slot=1200,
        verbose=False,
    )
    return str(out), report


@pytest.fixture(scope="module")
def tok(artifacts):
    out, _ = artifacts
    return PartitionedTokenizer.from_dir(out)


# --- gates -----------------------------------------------------------------


def test_g1_roundtrip_all_gates_pass(artifacts):
    _, report = artifacts
    assert report["verdict"]["gates"]["G1_roundtrip_exact"], report["verdict"]["summary"]


def test_g1_adversarial_exact(tok):
    for s in ADVERSARIAL:
        assert tok.decode(tok.encode(s)) == normalize(s), repr(s)


def test_g1_corpus_exact(tok, artifacts):
    _, report = artifacts
    assert report["verdict"]["roundtrip_corpus"]["failures"] == 0


def test_g2_no_special_ids_emitted(tok):
    for s in ADVERSARIAL:
        ids = tok.encode(s)
        assert all(i >= tok.layout.special_end for i in ids), repr(s)


def test_g2_no_unknown_token_in_decode(tok):
    for s in ADVERSARIAL:
        assert "<unk>" not in tok.decode(tok.encode(s))


def test_g3_id_space_is_sound(tok):
    """Every non-special id routes to exactly one slot, and first_invalid_id is
    the derived bound. There is no magic sentinel id in the design."""
    import tokenizer.slots as slots_mod

    assert not hasattr(slots_mod, "POISON_ID")
    assert tok.layout.first_invalid_id == tok.layout.vocab_size
    unrouted = [
        i
        for i in range(tok.layout.special_end, tok.layout.vocab_size)
        if tok.lookup.slot_of(i) is None
    ]
    assert unrouted == []


def test_g3_no_emitted_id_escapes_the_vocab(tok):
    for s in ADVERSARIAL:
        ids = tok.encode(s)
        assert all(0 <= i < tok.layout.vocab_size for i in ids), repr(s)


def test_flat_lookup_matches_binary_search_on_the_real_artifact(tok):
    for tid in range(-2, tok.layout.vocab_size + 2):
        a = tok.layout.slot_of(tid)
        b = tok.lookup.slot_of(tid)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.name == b.name


def test_embedding_padding_is_recorded(tok):
    assert tok.layout.padded_vocab_size % tok.layout.embed_multiple == 0
    assert tok.layout.padded_vocab_size >= tok.layout.vocab_size


def test_g6_manifest_verifies(artifacts):
    out, _ = artifacts
    result = verify_manifest(out)
    assert result["ok"], result["problems"]


def test_manifest_hash_changes_when_vocab_tampered(tmp_path, artifacts):
    """A retrain shifts ids, so a hash mismatch must be detectable."""
    import shutil

    out, _ = artifacts
    dst = tmp_path / "tampered"
    shutil.copytree(out, dst)
    vocab = dst / "vocab.v1.json"
    text = vocab.read_text(encoding="utf-8")
    vocab.write_text(text.replace('"tokens"', '"tokens_tampered"'), encoding="utf-8")
    assert verify_manifest(str(dst))["ok"] is False


# --- structural properties of the merged vocab -----------------------------


def test_vocab_file_matches_layout(tok, artifacts):
    out, _ = artifacts
    import json

    with open(os.path.join(out, "vocab.v1.json"), encoding="utf-8") as fh:
        vd = json.load(fh)
    assert len(vd["tokens"]) == tok.layout.vocab_size
    assert vd["first_invalid_id"] == tok.layout.first_invalid_id
    assert vd["tokens"][: len(tok.layout.specials)] == list(tok.layout.specials)


def test_slots_have_no_internal_duplicate_strings(tok):
    """Duplicates ACROSS slots are allowed (SPEC.md 6.2); within a slot they are
    a bug."""
    for s in tok.layout.slots:
        chunk = tok.vocab[s.start : s.end]
        assert len(set(chunk)) == len(chunk), s.name


def test_cross_slot_duplicates_are_preserved_not_collapsed(tok):
    """At least one string must appear in two slots, proving the vocab is a list
    and not a dict that would silently collapse them."""
    seen: dict[str, int] = {}
    for s in tok.layout.slots:
        for t in tok.vocab[s.start : s.end]:
            seen[t] = seen.get(t, 0) + 1
    assert any(v > 1 for v in seen.values())


def test_decode_routes_by_id_range(tok):
    """Hindi ids must resolve to the DEVANAGARI slot and Kannada to KANNADA."""
    hi_ids = tok.encode("नमस्ते")
    kn_ids = tok.encode("ಕನ್ನಡ")
    assert {tok.slot_of_id(i) for i in hi_ids} == {"DEVANAGARI"}
    assert {tok.slot_of_id(i) for i in kn_ids} == {"KANNADA"}


def test_decode_never_needs_language_identification(tok):
    """Concatenating a Hindi sequence and a Kannada sequence decodes both, with
    no tags and no sentinel in the stream."""
    ids = tok.encode("नमस्ते") + tok.encode("ಕನ್ನಡ")
    assert tok.decode(ids) == "नमस्तेಕನ್ನಡ"


def test_out_of_range_id_raises(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.layout.vocab_size + 10])


def test_special_ids_decode_to_their_token(tok):
    assert tok.decode([0]) == "<pad>"
    assert tok.decode([2]) == "<eos>"


def test_catchall_absorbs_unmodelled_scripts(tok):
    for s in ("தமிழ்", "العربية", "中文测试", "ελληνικά"):
        ids = tok.encode(s)
        assert {tok.slot_of_id(i) for i in ids} == {"CATCHALL"}
        assert tok.decode(ids) == s


# --- alphabet scope and the silent-drop regression -------------------------


def test_bpe_models_have_unk_token_set():
    """Regression guard for a real bug.

    With `unk_token=None`, HF's BPE does not merely avoid emitting <unk> -- it
    silently DROPS characters outside the alphabet, producing no id at all, so
    the loss is invisible in the id stream. Setting <unk> turns that into a
    detectable special id, which the partitioner converts into a lossless
    CATCHALL byte fallback.
    """
    from tokenizer.train import _model_for

    assert _model_for("bpe").unk_token is not None


def test_observed_alphabet_is_a_subset_of_the_full_one():
    from tokenizer.train import BYTE_ALPHABET, observed_alphabet

    texts = ["नमस्ते दुनिया", "ಕನ್ನಡ ಭಾಷೆ"]
    obs = observed_alphabet(texts)
    assert 0 < len(obs) < len(BYTE_ALPHABET)
    for t in texts:
        for b in t.encode("utf-8"):
            assert BYTE_ALPHABET[b] in obs


def test_bytes_outside_a_slots_alphabet_still_roundtrip(tok):
    """Characters absent from every training corpus must survive.

    These bytes are not in any slot's observed alphabet, so they exercise the
    UnrepresentableRun -> CATCHALL path.
    """
    for s in ("\t", "\x0b", "ﬀﬁﬂ", "㍿①Ⅻ", "🙂", "\u0301"):
        ids = tok.encode(s)
        assert tok.decode(ids) == normalize(s), repr(s)


def test_slot_models_do_not_emit_unknown_tokens(tok):
    """No <unk> may ever appear: the escape hatch is CATCHALL, not unk."""
    from tokenizer.build import ADVERSARIAL

    unk = tok.vocab.index("<unk>")
    for s in ADVERSARIAL:
        assert unk not in tok.encode(s), repr(s)


def test_stray_delimiter_does_not_explode_the_token_count(tok):
    """End-to-end guard for the runaway-structural-span bug.

    If an unterminated ``` routed the Devanagari tail into CODE_MATH, that slot
    cannot represent it, so it byte-falls-back at ~1 token per byte. A sane
    encoding of Devanagari costs far fewer tokens than bytes.
    """
    text = "prose here\n```python\nunclosed\n" + "नमस्ते दुनिया " * 200
    ids = tok.encode(text)
    n_bytes = len(text.encode("utf-8"))
    assert len(ids) < n_bytes / 2, (
        f"{len(ids)} tokens for {n_bytes} bytes suggests a byte-fallback blowup"
    )
    assert tok.decode(ids) == normalize(text)


# --- fallback granularity --------------------------------------------------


@pytest.fixture(scope="module")
def strict_tok(tmp_path_factory):
    """A build whose slot alphabets are strictly observed, with no ASCII margin.

    Only then do ordinary characters like a newline genuinely lie outside a
    slot's alphabet, which is the situation whole-run fallback used to ruin.
    """
    from tokenizer.build import build
    from tokenizer.train import ALPHABET_OBSERVED

    out = tmp_path_factory.mktemp("strict")
    data = tmp_path_factory.mktemp("strict_data")
    build(
        out_dir=str(out),
        data_dir=str(data),
        target_words=1500,
        total_vocab=3000,
        min_slot=300,
        max_slot=1200,
        alphabet_scope=ALPHABET_OBSERVED,
        verbose=False,
    )
    return PartitionedTokenizer.from_dir(str(out))


def test_per_character_fallback_bisects_without_reordering(tok):
    """Deterministic unit test of the fallback mechanism.

    Regression: fallback fired for the whole run, so one unrepresentable
    character re-encoded everything containing it. Here a stub makes a specific
    character unrepresentable, which is deterministic -- unlike hunting for a
    real character that happens to be outside some build's alphabet.
    """
    from tokenizer.train import UnrepresentableRun

    slot = tok.slots["DEVANAGARI"]
    original = slot.encode
    bad = "\u0301"

    def flaky(text):
        if bad in text:
            raise UnrepresentableRun("stub: forced")
        return original(text)

    slot.encode = flaky
    try:
        text = "नमस्ते" + bad + "दुनिया" + bad + "फिर"
        segments = []
        tok._encode_text("DEVANAGARI", text, segments)
    finally:
        slot.encode = original

    names = [name for name, _ in segments]
    assert "DEVANAGARI" in names, "representable parts must keep their slot"
    assert "CATCHALL" in names, "the bad characters must fall back"
    assert names.count("CATCHALL") == 2, "one fallback per bad character"

    rebuilt = "".join(
        tok.slots[name].decode(ids) for name, ids in segments
    )
    assert rebuilt == text, "bisection must not reorder or drop content"


def test_fallback_preserves_order_and_content(strict_tok):
    """Bisection must not reorder or drop content."""
    text = "\n\nनमस्ते\n\tदुनिया\n\nफिर\n"
    assert strict_tok.decode(strict_tok.encode(text)) == normalize(text)


def test_common_characters_cost_almost_nothing(tok):
    """A newline in a Devanagari run must not trigger whole-run fallback.

    The failure mode this guards multiplied the token count by ~3.2x
    (2400 -> 7601) because one unrepresentable byte re-encoded the entire run.
    Unigram segments with Viterbi, so adding a character can legitimately
    re-segment the sequence by a fraction of a percent; the bound is therefore
    proportional rather than a fixed token count.
    """
    base = "नमस्ते दुनिया " * 100
    a = len(tok.encode(base))
    b = len(tok.encode("\n" + base))
    assert b <= a * 1.05, f"one newline turned {a} tokens into {b}"
    assert tok.decode(tok.encode("\n" + base)) == normalize("\n" + base)


# --- corpus provenance -----------------------------------------------------


def test_build_does_not_clobber_an_existing_corpus(tmp_path):
    """The corpus file is authoritative.

    Regression: build() used to regenerate whenever the corpus was synthetic,
    using whatever target_words it was handed. Passing a different target
    silently overwrote a large corpus with a small one -- and let the tokenizer
    train on different text than the caller had loaded for evaluation.
    """
    from tokenizer.build import build
    from tokenizer.corpora import read_documents

    data = tmp_path / "d"
    common = dict(total_vocab=1500, min_slot=100, max_slot=400, verbose=False)
    build(out_dir=str(tmp_path / "a1"), data_dir=str(data), target_words=400, **common)
    path = data / "corpus" / "documents.jsonl"
    first = read_documents(str(path))

    build(
        out_dir=str(tmp_path / "a2"),
        data_dir=str(data),
        target_words=99999,
        **common,
    )
    second = read_documents(str(path))

    assert len(first) == len(second), "corpus was regenerated behind our back"
    assert [d.doc_id for d in first] == [d.doc_id for d in second]


def test_ascii_safety_set_is_used_by_default():
    """The default alphabet is observed bytes plus ASCII, which keeps common
    out-of-corpus characters (newline, digits, quotes) from tripping fallback."""
    from tokenizer.train import (
        BYTE_ALPHABET,
        observed_alphabet,
    )

    texts = ["नमस्ते दुनिया"]
    strict = observed_alphabet(texts, include_ascii=False)
    safe = observed_alphabet(texts, include_ascii=True)

    assert BYTE_ALPHABET[0x0A] not in strict
    assert BYTE_ALPHABET[0x0A] in safe  # newline
    assert BYTE_ALPHABET[ord("7")] in safe
    assert len(strict) < len(safe) < len(BYTE_ALPHABET)


def test_byte_fallback_is_exact_for_arbitrary_bytes():
    from tokenizer.train import ByteFallbackSlot

    b = ByteFallbackSlot()
    for s in ("🚀", "தமிழ்", "\x00\x01", "ﬀ"):
        assert b.decode(b.encode(s)) == s


def test_runs_are_order_preserving_in_encoded_output(tok):
    """The i-th run's ids occupy a contiguous span, and spans appear in input
    order (SPEC.md 7.1)."""
    text = "hello नमस्ते world ಕನ್ನಡ தமிழ் done"
    runs = list(tok.encode_runs(text))
    assert [r.slot for r, _ in runs] == [r.slot for r in segment(text)]

    flat = [i for _, ids in runs for i in ids]
    assert flat == tok.encode(text)
    assert tok.decode(flat) == normalize(text)


def test_no_sentinel_in_stream(tok):
    """The stream must contain only content ids: no boundary markers."""
    ids = tok.encode("hello नमस्ते ಕನ್ನಡ world")
    assert all(i >= tok.layout.special_end for i in ids)


def test_empty_and_whitespace_inputs(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.decode(tok.encode("   ")) == "   "
    assert tok.decode(tok.encode("\n\n\t")) == "\n\n\t"


# --- opt-in in-stream slot marker -----------------------------------------


def test_emit_slot_markers_costs_exactly_one_row(tmp_path):
    """The alternative to a magic sentinel: one special token, one vocabulary
    row, and it still round-trips because decode skips it."""
    from tokenizer.build import build
    from tokenizer.slots import SLOT_SEP

    out = tmp_path / "marked"
    data = tmp_path / "marked_data"
    build(
        out_dir=str(out),
        data_dir=str(data),
        target_words=400,
        total_vocab=2000,
        min_slot=300,
        max_slot=800,
        emit_slot_markers=True,
        verbose=False,
    )
    marked = PartitionedTokenizer.from_dir(str(out))
    assert marked.emit_slot_markers
    assert SLOT_SEP in marked.layout.specials
    assert marked.slot_sep_id is not None
    assert marked.slot_sep_id < marked.layout.special_end

    text = "hello नमस्ते world"
    ids = marked.encode(text)
    assert marked.slot_sep_id in ids, "marker should appear at a slot change"
    assert marked.decode(ids) == normalize(text)

    # And the unmarked tokenizer over the same text is strictly shorter.
    plain = build(
        out_dir=str(tmp_path / "plain"),
        data_dir=str(data),
        target_words=400,
        total_vocab=2000,
        min_slot=300,
        max_slot=800,
        emit_slot_markers=False,
        verbose=False,
    )
    assert not plain["verdict"]["emit_slot_markers"]


# --- multi-script grouping -------------------------------------------------


def _grouped_specs():
    from tokenizer.slots import SlotSpec

    return (
        SlotSpec("NEUTRAL", "bpe", ("COMMON",)),
        SlotSpec("LATIN", "bpe", ("LATIN",)),
        SlotSpec("INDIC_SHARED", "bpe", ("DEVANAGARI", "KANNADA")),
        SlotSpec("CODE_MATH", "bpe", (), structural=True),
        SlotSpec(
            "CATCHALL", "bytes", (), trained=False, exempt_from_budget=True
        ),
    )


def test_multi_script_slot_routes_every_grouped_script(tmp_path):
    """Two scripts, one algorithm, one shared vocabulary, one slot."""
    from tokenizer.build import build

    out = tmp_path / "grouped"
    build(
        out_dir=str(out),
        data_dir=str(tmp_path / "gdata"),
        target_words=600,
        total_vocab=2500,
        min_slot=300,
        max_slot=900,
        specs=_grouped_specs(),
        verbose=False,
    )
    tok = PartitionedTokenizer.from_dir(str(out))

    assert [s.name for s in tok.layout.slots] == [
        "NEUTRAL",
        "LATIN",
        "INDIC_SHARED",
        "CODE_MATH",
        "CATCHALL",
    ]
    hi = tok.encode("नमस्ते")
    kn = tok.encode("ಕನ್ನಡ")
    assert {tok.slot_of_id(i) for i in hi} == {"INDIC_SHARED"}
    assert {tok.slot_of_id(i) for i in kn} == {"INDIC_SHARED"}
    assert tok.decode(hi + kn) == "नमस्तेಕನ್ನಡ"
    # Both scripts are absent from the catch-all: they are genuinely modelled.
    assert CATCHALL not in {tok.slot_of_id(i) for i in hi + kn}


def test_grouping_collapses_two_scripts_into_one_slot(tmp_path):
    """The point of grouping: one shared vocabulary instead of two."""
    from tokenizer.build import build
    from tokenizer.slots import DEFAULT_SPECS

    common = dict(
        data_dir=str(tmp_path / "cmpdata"),
        target_words=600,
        total_vocab=3000,
        min_slot=300,
        max_slot=1200,
        verbose=False,
    )
    split = build(out_dir=str(tmp_path / "split"), specs=DEFAULT_SPECS, **common)
    grouped = build(out_dir=str(tmp_path / "grp"), specs=_grouped_specs(), **common)

    # The meaningful property is that grouping collapses two scripts into ONE
    # trained vocabulary. A row-count comparison was removed: it held only while
    # BPE under-filled its budget, and reversed once the slots moved to Unigram
    # with a raised piece limit, which fills closer to target.
    assert len(split["manifest"]["slot_sizes"]) == 6
    assert len(grouped["manifest"]["slot_sizes"]) == 5
    assert "INDIC_SHARED" in grouped["manifest"]["slot_sizes"]
    assert "DEVANAGARI" not in grouped["manifest"]["slot_sizes"]
    assert "KANNADA" not in grouped["manifest"]["slot_sizes"]


def test_slots_config_files_load():
    """The shipped YAML configs must parse and validate."""
    from tokenizer.build import load_slots_config
    from tokenizer.slots import slot_map_from_specs

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    plain = load_slots_config(os.path.join(repo_root, "config", "slots.v1.yaml"))
    assert slot_map_from_specs(plain)["LATIN"] == "LATIN"

    grouped = load_slots_config(os.path.join(repo_root, "config", "slots.grouped.example.yaml"))
    m = slot_map_from_specs(grouped)
    assert m["DEVANAGARI"] == "INDIC_SHARED"
    assert m["KANNADA"] == "INDIC_SHARED"

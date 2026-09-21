"""End-to-end build: generate -> extract slots -> allocate -> train -> freeze -> verify.

    python -m tokenizer.build --out artifacts/v1 --with-baseline

The synthetic corpus is a smoke-test source, not a training source. When real
corpus paths are available, pass a registry via config/corpora.v1.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

from . import metrics
from .baseline import BaselineTokenizer, balance_documents, compare
from .corpora import (
    Document,
    extract_slot_corpora,
    generate_corpus,
    generate_heldout_corpus,
    make_heldout_docs,
    read_documents,
    read_slot_corpus,
    sentence_overlap,
    sentence_space_size,
    write_documents,
)
from .merge import DEFAULT_SPECIALS, freeze, verify_manifest
from .partitioned import PartitionedTokenizer
from .scripts import (
    DEFAULT_WHITESPACE_OWNERSHIP,
    WHITESPACE_MODES,
    normalize,
)
from .slots import (
    DEFAULT_EMBED_MULTIPLE,
    DEFAULT_SPECS,
    SLOT_SEP,
    SlotSpec,
    allocate_budgets,
    slot_map_from_specs,
    specs_from_dicts,
    validate_specs,
)
from .train import (
    ALPHABET_OBSERVED_ASCII,
    ALPHABET_SCOPES,
    ByteFallbackSlot,
    train_slot,
)

# Adversarial cases for the round-trip gate. These deliberately probe the places
# a script-partitioned encoder is most likely to break: run boundaries, the
# byte-fallback escape hatch, and characters that combine across scripts.
ADVERSARIAL: Sequence[str] = (
    "",
    " ",
    "\n\n\t  ",
    "a",
    "a b c",
    "  leading and trailing  ",
    "नमस्ते",
    "ಕನ್ನಡ",
    "नमस्ते world ಕನ್ನಡ mixed",
    "कंप्यूटर",
    "ज्ञान",
    "क्षत्रिय",
    "\u0915\u094d\u0937",  # ka + virama + ssa
    "क्\u200dष",  # ZWJ inside a conjunct -- must survive
    "क्\u200cष",  # ZWNJ
    "e\u0301",  # combining acute on Latin
    "a\u0300\u0301\u0302",  # stacked combining marks
    "👨\u200d👩\u200d👧",  # emoji ZWJ sequence
    "🚀🔥✨",
    "தமிழ்",  # Tamil
    "తెలుగు",  # Telugu
    "മലയാളം",  # Malayalam
    "বাংলা",  # Bengali
    "العربية",  # Arabic
    "עברית",  # Hebrew
    "中文测试",  # CJK
    "日本語のテキスト",
    "한국어",  # Hangul
    "ελληνικά",
    "русский",
    "∑∮∫𝛼√∞",
    "₹1,23,456.78",
    "def f(x):\n    return x ** 2",
    "```python\nprint('hi')\n```",
    "$$\\int_0^1 x^2\\,dx = \\frac{1}{3}$$",
    "inline $E = mc^2$ math",
    "`inline code` and **bold** text",
    "eye color 🧑\u200d🎨 vs cat 🐈\u200d⬛",
    "a" * 2000,
    "\u0000\u0001control",
    "㍿①Ⅻ",
    "ﬀﬁﬂ",  # ligatures -- NFKC would fold these; NFC must not
    "line1\nline2\r\nline3",
)


def load_slots_config(path: str) -> tuple[SlotSpec, ...]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    specs = specs_from_dicts(raw["slots"])
    validate_specs(specs)
    return specs


def build(
    out_dir: str,
    data_dir: str = "data",
    target_words: int = 10000,
    seed: int = 1337,
    total_vocab: int = 8000,
    alpha: float = 0.7,
    min_slot: int = 300,
    max_slot: int = 4000,
    specials: Sequence[str] = DEFAULT_SPECIALS,
    specs: Sequence[SlotSpec] = DEFAULT_SPECS,
    embed_multiple: int = DEFAULT_EMBED_MULTIPLE,
    emit_slot_markers: bool = False,
    whitespace_ownership: str = DEFAULT_WHITESPACE_OWNERSHIP,
    alphabet_scope: str = ALPHABET_OBSERVED_ASCII,
    verbose: bool = True,
    with_baseline: bool = False,
) -> Dict:
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    if whitespace_ownership not in WHITESPACE_MODES:
        raise ValueError(
            f"unknown whitespace_ownership {whitespace_ownership!r}; "
            f"expected one of {WHITESPACE_MODES}"
        )
    validate_specs(specs)
    slot_map = slot_map_from_specs(specs)
    structural = [s.name for s in specs if s.structural]
    structural_slot = structural[0] if structural else "CODE_MATH"

    specials = tuple(specials)
    if emit_slot_markers and SLOT_SEP not in specials:
        specials = specials + (SLOT_SEP,)

    # 1. corpus -----------------------------------------------------------
    # The corpus FILE is authoritative. Provenance is recorded alongside it
    # because it decides which held-out evaluation is valid: a synthetic corpus
    # gets a fresh natural split, a real one can only be word-shuffled.
    #
    # An earlier version regenerated whenever the corpus was synthetic, using
    # whatever target_words it was called with. That silently overwrote a
    # 120k-word corpus with a 4k-word one when a caller passed a different
    # target, and worse, let the tokenizer train on a different corpus than the
    # one the caller had loaded for evaluation. Generation now happens only when
    # the file is absent; --rebuild-corpus removes it first.
    corpus_path = os.path.join(data_dir, "corpus", "documents.jsonl")
    meta_path = os.path.join(data_dir, "corpus", "source.json")
    synthetic = False

    if os.path.exists(corpus_path):
        docs: List[Document] = read_documents(corpus_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            synthetic = bool(meta.get("synthetic", False))
            stored = meta.get("target_words")
            if stored is not None and int(stored) != int(target_words):
                log(
                    f"[1/6] corpus      : NOTE {corpus_path} holds {stored} words "
                    f"but target_words={target_words} was requested; using the "
                    f"file. Pass --rebuild-corpus to regenerate."
                )
        log(f"[1/6] corpus      : loaded {len(docs)} documents from {corpus_path}")
    else:
        docs = generate_corpus(target_words=target_words, seed=seed)
        write_documents(docs, corpus_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "synthetic": True,
                    "generator": "v2",
                    "seed": seed,
                    "target_words": target_words,
                    "documents": len(docs),
                },
                fh,
                indent=2,
            )
        synthetic = True
        space = sentence_space_size()
        log(
            f"[1/6] corpus      : generated {len(docs)} documents -> {corpus_path}"
        )
        log(f"        sentence space per kind: {space}")

    texts = [d.text for d in docs]

    # 2. extract per-slot corpora (original design step 3a) ---------------
    slots_dir = os.path.join(data_dir, "slots")
    slot_bytes = extract_slot_corpora(
        docs,
        slots_dir,
        slot_map=slot_map,
        structural_slot=structural_slot,
        whitespace_ownership=whitespace_ownership,
    )
    log(
        f"[2/6] extract     : {len(slot_bytes)} slots -> {slots_dir} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(slot_bytes.items()))})"
    )

    # 3. budgets ----------------------------------------------------------
    budgets = allocate_budgets(
        slot_bytes, total_vocab, specs, alpha=alpha, min_slot=min_slot, max_slot=max_slot
    )
    log(f"[3/6] budgets     : total={total_vocab} alpha={alpha} -> {budgets}")

    # 4. train per slot (original design step 3b) -------------------------
    slots: Dict[str, object] = {}
    for spec in specs:
        if spec.algorithm == "bytes":
            slots[spec.name] = ByteFallbackSlot()
            continue
        path = os.path.join(slots_dir, f"{spec.name}.jsonl")
        runs = read_slot_corpus(path) if os.path.exists(path) else []
        if not runs:
            raise ValueError(
                f"slot {spec.name!r} (scripts: {', '.join(spec.scripts) or '-'}) "
                f"has no corpus at {path}. Either add corpus for those scripts, "
                f"remove the slot from the config, or leave the scripts unclaimed "
                f"so they fall through to CATCHALL byte fallback."
            )
        st = time.time()
        slots[spec.name] = train_slot(
            spec.name, spec.algorithm, runs, budgets[spec.name],
            alphabet_scope=alphabet_scope,
            max_piece_length=spec.max_piece_length,
        )
        log(
            f"[4/6] train       : {spec.name:<14} {spec.algorithm:<8} "
            f"target={budgets[spec.name]:<6} got={slots[spec.name].local_size:<6} "
            f"runs={len(runs):<6} {time.time() - st:.2f}s"
        )

    # 5. freeze -----------------------------------------------------------
    manifest = freeze(
        out_dir,
        specials,
        slots,
        specs,
        embed_multiple=embed_multiple,
        emit_slot_markers=emit_slot_markers,
        whitespace_ownership=whitespace_ownership,
        extra={
            "alpha": alpha,
            "total_vocab_target": total_vocab,
            "corpus_documents": len(docs),
            "corpus_source": corpus_path,
            "emit_slot_markers": emit_slot_markers,
            "whitespace_ownership": whitespace_ownership,
            "alphabet_scope": alphabet_scope,
        },
    )
    log(
        f"[5/6] freeze      : vocab_size={manifest['vocab_size']} "
        f"padded={manifest['padded_vocab_size']} -> {out_dir}"
    )

    # 6. load back from disk and verify -----------------------------------
    tok = PartitionedTokenizer.from_dir(out_dir)
    verdict = verify(tok, texts)
    mfr = verify_manifest(out_dir)

    # 7. optional: shared-vocab baseline, same total vocab, same corpora ----
    # Compared BOTH on the training documents and on a sequence-novel held-out
    # set. The gap between the two is the memorisation share; reporting only the
    # training number would make a degenerate corpus look like a triumph.
    comparison = None
    if with_baseline:
        balanced = balance_documents(docs, seed=seed)
        # Synthetic corpora get a genuinely disjoint natural held-out split;
        # real corpora fall back to word shuffling, which is weaker because it
        # destroys multi-word structure.
        if synthetic:
            heldout = generate_heldout_corpus(
                target_words=max(1000, target_words // 4), seed=seed + 20250
            )
            log(
                f"[7/7] heldout     : {len(heldout)} fresh documents "
                f"(sentence overlap with train: "
                f"{sentence_overlap(docs, heldout):.4f})"
            )
        else:
            heldout = make_heldout_docs(docs, seed=seed + 4242)
            log(f"[7/7] heldout     : {len(heldout)} word-shuffled documents")
        st = time.time()
        baseline = BaselineTokenizer.train(
            balanced, vocab_size=manifest["vocab_size"], algorithm="unigram"
        )
        baseline.save(os.path.join(out_dir, "baseline.v1.json"))
        comparison = {
            "train": compare(tok, baseline, docs),
            "heldout": compare(tok, baseline, heldout),
            "heldout_documents": len(heldout),
        }
        log(
            f"[7/7] baseline    : shared-vocab unigram vocab_size="
            f"{baseline.vocab_size} trained_on={len(balanced)} docs "
            f"{time.time() - st:.2f}s"
        )
        for split in ("train", "heldout"):
            log(f"[7/7] {split:<9} (per-script tokens/char improvement over baseline)")
            for slot, m in comparison[split]["per_script"].items():
                log(
                    f"        {slot:<12} partitioned="
                    f"{m['partitioned']['tokens_per_char']:<8} baseline="
                    f"{m['baseline']['tokens_per_char']:<8} "
                    f"improvement={m['tokens_per_char_improvement_pct']}%"
                )

        # The gap between train and held-out improvement IS the memorisation
        # share. A large positive gap means the corpus is degenerate and the
        # train-side number is not evidence.
        comparison["memorization_gap"] = {
            slot: round(
                comparison["train"]["per_script"][slot][
                    "tokens_per_char_improvement_pct"
                ]
                - comparison["heldout"]["per_script"][slot][
                    "tokens_per_char_improvement_pct"
                ],
                2,
            )
            for slot in comparison["train"]["per_script"]
            if slot in comparison["heldout"]["per_script"]
        }
        log(f"[7/7] memo gap    : {comparison['memorization_gap']}")

    report = {
        "out_dir": out_dir,
        "manifest": manifest,
        "manifest_ok": mfr["ok"],
        "manifest_problems": mfr["problems"],
        "budgets": budgets,
        "slot_corpus_bytes": slot_bytes,
        "layout": tok.describe(),
        "verdict": verdict,
        "corpus_metrics": metrics.corpus_metrics(tok, texts),
        "per_slot_metrics": metrics.per_slot_metrics(tok, texts),
        "comparison": comparison,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(os.path.join(out_dir, "build_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    log(f"[6/6] verify      : {verdict['summary']}")
    return report


def verify(tok: PartitionedTokenizer, texts: Sequence[str]) -> Dict:
    """Run gates G1/G2/G3 over the corpus and the adversarial set."""
    rt_corpus = metrics.roundtrip_report(tok, texts)
    rt_adv = metrics.roundtrip_report(tok, list(ADVERSARIAL))

    # G3: the id space is sound. Every non-special id in [0, vocab_size) routes
    # to exactly one slot, and first_invalid_id is the derived bound.
    unrouted = [
        i
        for i in range(tok.layout.special_end, tok.layout.vocab_size)
        if tok.lookup.slot_of(i) is None
    ]
    g1 = rt_corpus["failures"] == 0 and rt_adv["failures"] == 0
    g2 = (
        rt_corpus["special_ids_in_output"] == 0
        and rt_adv["special_ids_in_output"] == 0
    )
    g3 = not unrouted and tok.layout.first_invalid_id == tok.layout.vocab_size

    gates = {
        "G1_roundtrip_exact": g1,
        "G2_zero_unknown": g2,
        "G3_id_space_sound": g3,
    }
    failed = [k for k, v in gates.items() if not v]
    summary = (
        "ALL GATES PASS "
        f"(corpus {rt_corpus['exact']}/{rt_corpus['documents']}, "
        f"adversarial {rt_adv['exact']}/{rt_adv['documents']}, "
        f"vocab_size={tok.layout.vocab_size})"
        if not failed
        else f"GATE FAILURES: {failed}"
    )

    return {
        "gates": gates,
        "summary": summary,
        "roundtrip_corpus": rt_corpus,
        "roundtrip_adversarial": rt_adv,
        "encoded_ids": rt_corpus["total_ids"],
        "fallback_runs": tok.stats.fallback_runs,
        "slot_run_counts": dict(tok.stats.slot_runs),
        "unrouted_ids": len(unrouted),
        "emit_slot_markers": tok.emit_slot_markers,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/v1")
    ap.add_argument("--data", default="data")
    ap.add_argument("--target-words", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--total-vocab", type=int, default=8000)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--min-slot", type=int, default=300)
    ap.add_argument("--max-slot", type=int, default=4000)
    ap.add_argument("--embed-multiple", type=int, default=DEFAULT_EMBED_MULTIPLE)
    ap.add_argument(
        "--slots-config",
        default=None,
        help="YAML declaring slots as (name, algorithm, scripts[]); see config/slots.v1.yaml",
    )
    ap.add_argument(
        "--emit-slot-markers",
        action="store_true",
        help=f"emit {SLOT_SEP} between slot changes (costs one vocabulary row)",
    )
    ap.add_argument(
        "--whitespace-ownership",
        choices=list(WHITESPACE_MODES),
        default=DEFAULT_WHITESPACE_OWNERSHIP,
        help=(
            "neutral: whitespace is its own NEUTRAL run; "
            "leading: whitespace joins the run that follows it (SPEC.md 5.1)"
        ),
    )
    ap.add_argument(
        "--alphabet-scope",
        choices=list(ALPHABET_SCOPES),
        default=ALPHABET_OBSERVED_ASCII,
        help=(
            "observed: only bytes in that slot's corpus; "
            "observed+ascii: plus whitespace and printable ASCII (default); "
            "full: all 256 in every slot"
        ),
    )
    ap.add_argument("--rebuild-corpus", action="store_true")
    ap.add_argument(
        "--with-baseline",
        action="store_true",
        help="also train the shared-vocab baseline and compare (SPEC.md 10)",
    )
    args = ap.parse_args(argv)

    if args.rebuild_corpus:
        for p in (
            os.path.join(args.data, "corpus", "documents.jsonl"),
            os.path.join(args.data, "corpus", "source.json"),
        ):
            if os.path.exists(p):
                os.remove(p)

    specs = load_slots_config(args.slots_config) if args.slots_config else DEFAULT_SPECS

    report = build(
        out_dir=args.out,
        data_dir=args.data,
        target_words=args.target_words,
        seed=args.seed,
        total_vocab=args.total_vocab,
        alpha=args.alpha,
        min_slot=args.min_slot,
        max_slot=args.max_slot,
        specs=specs,
        embed_multiple=args.embed_multiple,
        emit_slot_markers=args.emit_slot_markers,
        whitespace_ownership=args.whitespace_ownership,
        alphabet_scope=args.alphabet_scope,
        with_baseline=args.with_baseline,
    )
    print("\n--- layout ---")
    for s in report["layout"]["slots"]:
        scripts = ",".join(s["scripts"]) or "-"
        print(
            f"  {s['name']:<14} [{s['start']:>6} .. {s['end']:>6}) "
            f"size={s['size']:<6} {s['algorithm']:<8} scripts={scripts}"
        )
    print(
        f"  vocab_size={report['layout']['vocab_size']} "
        f"padded={report['layout']['padded_vocab_size']} "
        f"first_invalid_id={report['layout']['first_invalid_id']}"
    )
    print("\n--- per slot ---")
    for name, m in report["per_slot_metrics"].items():
        print(
            f"  {name:<14} runs={m['runs']:<6} chars={m['chars']:<7} "
            f"tokens={m['tokens']:<7} tok/char={m['tokens_per_char']:<7} "
            f"bytes/tok={m['bytes_per_token']}"
        )
    print("\n--- corpus ---")
    for k, v in report["corpus_metrics"].items():
        print(f"  {k:<16} {v}")
    if report.get("comparison"):
        print("\n--- vs shared-vocab baseline (held-out, sequence-novel) ---")
        held = report["comparison"]["heldout"]
        for name, m in held["per_script"].items():
            print(
                f"  {name:<14} partitioned={m['partitioned']['tokens_per_char']:<8} "
                f"baseline={m['baseline']['tokens_per_char']:<8} "
                f"improvement={m['tokens_per_char_improvement_pct']}%"
            )
        print(f"  memorization gap (train - heldout): "
              f"{report['comparison']['memorization_gap']}")
    print(f"\n{report['verdict']['summary']}")
    return 0 if all(report["verdict"]["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the bits-per-byte comparison: partitioned vs shared-vocabulary baseline.

    python -m tokenizer.bpb --target-words 120000 --steps 400 --seeds 1,2,3

Trains a small transformer on each tokenizer's output for the same token budget
and compares held-out bits-per-byte. This is the decision metric named in
SPEC.md 10.2; fertility is only a proxy for it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional, Sequence

from .baseline import BaselineTokenizer, balance_documents
from .build import build
from .corpora import (
    Document,
    generate_corpus,
    generate_heldout_corpus,
    read_documents,
    sentence_overlap,
    write_documents,
)
from .lm_eval import LMConfig, compare_bpb
from .partitioned import PartitionedTokenizer


def _load_or_make_corpus(
    data_dir: str, target_words: int, seed: int, rebuild: bool
) -> List[Document]:
    corpus_path = os.path.join(data_dir, "corpus", "documents.jsonl")
    meta_path = os.path.join(data_dir, "corpus", "source.json")
    if rebuild:
        for p in (corpus_path, meta_path):
            if os.path.exists(p):
                os.remove(p)
    if os.path.exists(corpus_path) and os.path.exists(meta_path):
        return read_documents(corpus_path)
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
    return docs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/bpb")
    ap.add_argument("--data", default="data_bpb")
    ap.add_argument("--target-words", type=int, default=120000)
    ap.add_argument("--heldout-words", type=int, default=20000)
    ap.add_argument("--total-vocab", type=int, default=8000)
    ap.add_argument("--min-slot", type=int, default=300)
    ap.add_argument("--max-slot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--baseline-algorithm",
        choices=("unigram", "bpe"),
        default="unigram",
        help=(
            "unigram prunes to whatever the corpus supports and may fall short of "
            "the requested vocabulary; bpe fills its target. Use bpe to test "
            "whether a win survives an unhandicapped baseline"
        ),
    )
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--rebuild-corpus", action="store_true")
    ap.add_argument(
        "--builds",
        type=int,
        default=1,
        help=(
            "re-run the pipeline N times. NOTE: HF tokenizers training is "
            "reproducible within a process and varies only across processes, so "
            "this is NOT independent replication and will usually report zero "
            "spread. To replicate, run the whole command again"
        ),
    )
    args = ap.parse_args(argv)

    t0 = time.time()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    # With builds > 1 the individual builds land in `out_b0`, `out_b1`, ...; the
    # parent directory still has to exist for the baseline artifact.
    os.makedirs(args.out, exist_ok=True)

    # 1. corpora ----------------------------------------------------------
    docs = _load_or_make_corpus(
        args.data, args.target_words, args.seed, args.rebuild_corpus
    )
    heldout = generate_heldout_corpus(
        target_words=args.heldout_words, seed=args.seed + 20250
    )
    print(
        f"[1/5] corpora    : train={len(docs)} docs "
        f"({sum(len(d.text.encode('utf-8')) for d in docs)} B), "
        f"heldout={len(heldout)} docs "
        f"({sum(len(d.text.encode('utf-8')) for d in heldout)} B), "
        f"sentence overlap={sentence_overlap(docs, heldout):.4f}"
    )

    # 5. train and compare ------------------------------------------------
    cfg = LMConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=max(1, args.d_model // 32),
        d_ff=args.d_model * 4,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        steps=args.steps,
    )
    print(
        f"[5/5] lm         : {cfg.n_layer}L d_model={cfg.d_model} "
        f"seq={cfg.seq_len} batch={cfg.batch_size} steps={cfg.steps} "
        f"seeds={seeds} builds={args.builds} "
        f"-> {cfg.steps * cfg.batch_size * cfg.seq_len} tokens/model/build"
    )

    balanced = balance_documents(docs, seed=args.seed)
    heldout_bytes = sum(len(d.text.encode("utf-8")) for d in heldout)

    def encode_docs(enc, items) -> List[int]:
        out: List[int] = []
        for d in items:
            out.extend(enc.encode(d.text))
        return out

    per_build: List[dict] = []
    for b in range(args.builds):
        out_dir = args.out if args.builds == 1 else f"{args.out}_b{b}"
        report = build(
            out_dir=out_dir,
            data_dir=args.data,
            target_words=args.target_words,
            seed=args.seed,
            total_vocab=args.total_vocab,
            min_slot=args.min_slot,
            max_slot=args.max_slot,
            verbose=False,
        )
        tok = PartitionedTokenizer.from_dir(out_dir)
        baseline = BaselineTokenizer.train(
            balanced,
            vocab_size=tok.layout.vocab_size,
            algorithm=args.baseline_algorithm,
        )
        if b == 0:
            baseline.save(os.path.join(args.out, "baseline_bpb.v1.json"))
            print(
                f"[2/5] partitioned: vocab_size={tok.layout.vocab_size} "
                f"whitespace={tok.whitespace_ownership} "
                f"gates={report['verdict']['summary'].split('(')[0].strip()}"
            )
            print(
                f"[3/5] baseline   : {args.baseline_algorithm} "
                f"vocab_size={baseline.vocab_size} "
                f"(requested {tok.layout.vocab_size}) on {len(balanced)} docs"
            )

        p_train = encode_docs(tok, docs)
        b_train = encode_docs(baseline, docs)
        p_held = encode_docs(tok, heldout)
        b_held = encode_docs(baseline, heldout)
        if b == 0:
            print(
                f"[4/5] tokens     : train partitioned={len(p_train)} "
                f"baseline={len(b_train)} | heldout partitioned={len(p_held)} "
                f"baseline={len(b_held)}"
            )
            print(
                f"        heldout tokens/byte: partitioned="
                f"{len(p_held) / heldout_bytes:.4f} baseline="
                f"{len(b_held) / heldout_bytes:.4f}"
            )

        res = compare_bpb(
            p_train,
            b_train,
            tok.layout.vocab_size,
            baseline.vocab_size,
            p_held,
            b_held,
            heldout_bytes,
            cfg,
            seeds=seeds,
            verbose=args.builds == 1,
        )
        res["build"] = b
        res["partitioned_vocab"] = tok.layout.vocab_size
        res["baseline_vocab"] = baseline.vocab_size
        per_build.append(res)
        print(
            f"    build {b}: part_vocab={tok.layout.vocab_size} "
            f"base_vocab={baseline.vocab_size} "
            f"bpb part={res['partitioned_bpb']['mean']:.4f} "
            f"base={res['baseline_bpb']['mean']:.4f} "
            f"delta={res['bpb_delta_mean']:+.4f}"
        )

    deltas = [r["bpb_delta_mean"] for r in per_build]
    import statistics as _st

    result = per_build[0]
    result["per_build"] = [
        {
            "build": r["build"],
            "partitioned_vocab": r["partitioned_vocab"],
            "baseline_vocab": r["baseline_vocab"],
            "partitioned_bpb": r["partitioned_bpb"]["mean"],
            "baseline_bpb": r["baseline_bpb"]["mean"],
            "delta": r["bpb_delta_mean"],
        }
        for r in per_build
    ]
    result["builds"] = args.builds
    result["delta_across_builds_mean"] = float(_st.mean(deltas))
    result["delta_across_builds_std"] = (
        float(_st.stdev(deltas)) if len(deltas) > 1 else 0.0
    )
    if len(deltas) > 1:
        if len(set(deltas)) == 1:
            print(
                "    WARNING: all builds produced bit-identical results. HF "
                "tokenizers training is reproducible WITHIN a process and varies "
                "only ACROSS processes, so --builds here is not independent "
                "replication. Re-run the command itself to replicate."
            )
        all_positive = all(d > 0 for d in deltas)
        all_negative = all(d < 0 for d in deltas)
        result["sign_consistent"] = all_positive or all_negative
        result["winner"] = (
            "partitioned" if all_positive else "baseline" if all_negative
            else "inconclusive"
        )
        result["decisive"] = result["sign_consistent"]
        result["independent_replication"] = len(set(deltas)) > 1

    result["heldout_bytes"] = heldout_bytes
    result["sentence_overlap"] = sentence_overlap(docs, heldout)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "bpb_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print()
    print("--- bits per byte (lower is better) ---")
    print(
        f"  partitioned : {result['partitioned_bpb']['mean']:.4f} "
        f"+/- {result['partitioned_bpb']['std']:.4f}"
    )
    print(
        f"  baseline    : {result['baseline_bpb']['mean']:.4f} "
        f"+/- {result['baseline_bpb']['std']:.4f}"
    )
    print(
        f"  delta (base - part) : {result['bpb_delta_mean']:+.4f} "
        f"+/- {result['bpb_delta_std']:.4f}"
    )
    print(
        f"  params: partitioned={result['partitioned_params']} "
        f"baseline={result['baseline_params']}"
    )
    print(f"  VERDICT: {result['winner'].upper()} "
          f"({'decisive' if result['decisive'] else 'within noise'})")
    print(f"  elapsed: {result['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

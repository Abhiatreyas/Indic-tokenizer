"""Command-line interface for Indic-Partitioned Tokenizer.

Commands:
    indic-tokenizer build       Build tokenizer from config
    indic-tokenizer encode      Encode text to token IDs
    indic-tokenizer decode      Decode token IDs to text
    indic-tokenizer shard       Pre-process & shard large corpora into binary .bin files
    indic-tokenizer benchmark   Run comprehensive multi-language benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")
from typing import List

from .build import build, load_slots_config
from .hf_tokenizer import IndicPartitionedTokenizer
from .partitioned import PartitionedTokenizer
from .pipeline import shard_corpus


def cmd_build(args: argparse.Namespace) -> None:
    specs = load_slots_config(args.slots_config) if args.slots_config else None
    res = build(
        out_dir=args.out,
        data_dir=args.data_dir,
        total_vocab=args.vocab,
        specs=specs if specs else None,
        with_baseline=args.with_baseline,
    )
    print(f"Build complete. Vocab size: {res['vocab_size']}, saved to: {args.out}")


def cmd_encode(args: argparse.Namespace) -> None:
    tok = PartitionedTokenizer.from_dir(args.tokenizer_dir)
    text = args.text
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    if not text:
        print("Error: No text provided.", file=sys.stderr)
        sys.exit(1)
    ids = tok.encode(text)
    if args.json:
        print(json.dumps({"tokens": len(ids), "ids": ids}))
    else:
        print(f"Tokens ({len(ids)}): {ids}")


def cmd_decode(args: argparse.Namespace) -> None:
    tok = PartitionedTokenizer.from_dir(args.tokenizer_dir)
    ids = [int(x) for x in args.ids]
    text = tok.decode(ids)
    print(text)


def cmd_shard(args: argparse.Namespace) -> None:
    shard_corpus(
        input_paths=args.inputs,
        output_dir=args.output,
        tokenizer_dir=args.tokenizer_dir,
        shard_size_tokens=args.shard_size,
        text_key=args.text_key,
        num_workers=args.workers,
        max_docs=args.max_docs,
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "..", "diagnostics", "run_full_benchmark.py"),
        "--tokenizer-dir", args.tokenizer_dir,
    ]
    subprocess.run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="indic-tokenizer",
        description="Script-Partitioned Multilingual Tokenizer for Indic & English LLMs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = subparsers.add_parser("build", help="Build tokenizer artifacts")
    p_build.add_argument("--out", required=True, help="Output artifact directory")
    p_build.add_argument("--slots-config", default=None, help="Path to YAML slots configuration")
    p_build.add_argument("--vocab", type=int, default=128000, help="Total target vocabulary size")
    p_build.add_argument("--data-dir", default="data", help="Corpus cache directory")
    p_build.add_argument("--with-baseline", action="store_true", help="Compare with shared baseline")
    p_build.set_defaults(func=cmd_build)

    # encode
    p_enc = subparsers.add_parser("encode", help="Encode text")
    p_enc.add_argument("text", nargs="?", default="", help="Text to encode")
    p_enc.add_argument("--tokenizer-dir", required=True, help="Tokenizer artifact directory")
    p_enc.add_argument("--json", action="store_true", help="Output JSON with token count")
    p_enc.set_defaults(func=cmd_encode)

    # decode
    p_dec = subparsers.add_parser("decode", help="Decode token IDs")
    p_dec.add_argument("ids", nargs="+", help="Token IDs to decode")
    p_dec.add_argument("--tokenizer-dir", required=True, help="Tokenizer artifact directory")
    p_dec.set_defaults(func=cmd_decode)

    # shard
    p_shard = subparsers.add_parser("shard", help="Tokenize and shard large JSONL corpora for pre-training")
    p_shard.add_argument("--inputs", nargs="+", required=True, help="Input JSONL/text files or globs")
    p_shard.add_argument("--output", required=True, help="Output directory for binary shards")
    p_shard.add_argument("--tokenizer-dir", required=True, help="Tokenizer artifact directory")
    p_shard.add_argument("--shard-size", type=int, default=100_000_000, help="Tokens per binary shard file")
    p_shard.add_argument("--text-key", default="text", help="JSON key containing text")
    p_shard.add_argument("--workers", type=int, default=None, help="Number of CPU worker processes")
    p_shard.add_argument("--max-docs", type=int, default=None, help="Maximum documents to process")
    p_shard.set_defaults(func=cmd_shard)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run multi-language benchmark")
    p_bench.add_argument("--tokenizer-dir", required=True, help="Tokenizer artifact directory")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""High-throughput multi-processing dataset tokenization and binary sharding pipeline.

Packs large text corpora (JSONL/text) into contiguous binary token arrays (.bin / .idx)
for pre-training foundation LLMs on PyTorch, LitGPT, and NanoGPT.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .partitioned import PartitionedTokenizer


def _tokenize_chunk(args: Tuple[List[str], str, int]) -> List[int]:
    """Worker function to tokenize a batch of text documents."""
    texts, tokenizer_dir, eos_id = args
    tok = PartitionedTokenizer.from_dir(tokenizer_dir)
    tokens: List[int] = []
    for text in texts:
        if not text or not text.strip():
            continue
        ids = tok.encode(text)
        tokens.extend(ids)
        if eos_id is not None:
            tokens.append(eos_id)
    return tokens


def stream_jsonl_texts(
    paths: Sequence[str], text_key: str = "text", max_docs: Optional[int] = None
) -> Iterator[str]:
    """Yield text strings from JSONL or raw text files."""
    doc_count = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{") and line.endswith("}"):
                    try:
                        record = json.loads(line)
                        text = record.get(text_key, "")
                        if not text and "content" in record:
                            text = record["content"]
                        elif not text and "target" in record:
                            text = record["target"]
                    except Exception:
                        text = line
                else:
                    text = line

                if text:
                    yield text
                    doc_count += 1
                    if max_docs is not None and doc_count >= max_docs:
                        return


def shard_corpus(
    input_paths: Sequence[str],
    output_dir: str,
    tokenizer_dir: str,
    shard_size_tokens: int = 100_000_000,
    text_key: str = "text",
    num_workers: Optional[int] = None,
    chunk_batch_size: int = 500,
    eos_id: int = 2,
    max_docs: Optional[int] = None,
    dtype: str = "auto",
) -> Dict[str, Any]:
    """Tokenize and shard text corpora into binary training shards (.bin)."""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    tok = PartitionedTokenizer.from_dir(tokenizer_dir)
    vocab_size = tok.layout.vocab_size

    if dtype == "auto":
        np_dtype = np.uint16 if vocab_size < 65536 else np.uint32
    elif dtype == "uint16":
        if vocab_size >= 65536:
            raise ValueError(f"vocab_size {vocab_size} >= 65536 requires uint32")
        np_dtype = np.uint16
    else:
        np_dtype = np.uint32

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    print("=" * 80)
    print(f"SHARDING CORPUS FOR PRE-TRAINING")
    print(f"  Tokenizer Dir    : {tokenizer_dir}")
    print(f"  Vocab Size       : {vocab_size}")
    print(f"  Token Dtype      : {np_dtype.__name__}")
    print(f"  Shard Size       : {shard_size_tokens:,} tokens")
    print(f"  CPU Workers      : {num_workers}")
    print(f"  Output Directory : {output_dir}")
    print("=" * 80)

    shard_idx = 0
    total_tokens = 0
    total_docs = 0
    current_shard_tokens: List[int] = []
    shards_manifest: List[Dict[str, Any]] = []

    text_iterator = stream_jsonl_texts(input_paths, text_key=text_key, max_docs=max_docs)

    def write_shard(tokens: List[int], s_idx: int) -> str:
        shard_fname = f"shard_{s_idx:05d}.bin"
        shard_path = os.path.join(output_dir, shard_fname)
        arr = np.array(tokens, dtype=np_dtype)
        with open(shard_path, "wb") as fh:
            fh.write(arr.tobytes())
        return shard_path

    # Multi-processing pool for tokenization
    with mp.Pool(num_workers) as pool:
        batch: List[str] = []
        batch_tasks: List[Tuple[List[str], str, int]] = []

        for text in text_iterator:
            batch.append(text)
            total_docs += 1
            if len(batch) >= chunk_batch_size:
                batch_tasks.append((batch, tokenizer_dir, eos_id))
                batch = []

            if len(batch_tasks) >= num_workers * 2:
                results = pool.map(_tokenize_chunk, batch_tasks)
                for res in results:
                    current_shard_tokens.extend(res)
                    total_tokens += len(res)

                    if len(current_shard_tokens) >= shard_size_tokens:
                        to_write = current_shard_tokens[:shard_size_tokens]
                        current_shard_tokens = current_shard_tokens[shard_size_tokens:]
                        spath = write_shard(to_write, shard_idx)
                        elapsed = time.time() - t0
                        speed = total_tokens / max(elapsed, 0.001)
                        print(
                            f"  [Shard {shard_idx:05d}] Saved {len(to_write):,} tokens -> {spath} "
                            f"| Total: {total_tokens / 1e6:.2f}M tok ({speed / 1e3:.1f}k tok/s)"
                        )
                        shards_manifest.append({
                            "shard_index": shard_idx,
                            "filename": os.path.basename(spath),
                            "num_tokens": len(to_write),
                            "dtype": np_dtype.__name__,
                        })
                        shard_idx += 1
                batch_tasks = []

        # Process any remaining tasks
        if batch:
            batch_tasks.append((batch, tokenizer_dir, eos_id))
        if batch_tasks:
            results = pool.map(_tokenize_chunk, batch_tasks)
            for res in results:
                current_shard_tokens.extend(res)
                total_tokens += len(res)

        # Write final partial shard
        if current_shard_tokens:
            spath = write_shard(current_shard_tokens, shard_idx)
            shards_manifest.append({
                "shard_index": shard_idx,
                "filename": os.path.basename(spath),
                "num_tokens": len(current_shard_tokens),
                "dtype": np_dtype.__name__,
            })
            shard_idx += 1

    elapsed = time.time() - t0
    report = {
        "tokenizer_dir": os.path.abspath(tokenizer_dir),
        "vocab_size": vocab_size,
        "dtype": np_dtype.__name__,
        "num_shards": len(shards_manifest),
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(total_tokens / max(elapsed, 0.001), 1),
        "shards": shards_manifest,
    }

    manifest_path = os.path.join(output_dir, "shards_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 80)
    print(f"SHARDING COMPLETE:")
    print(f"  Total Tokens     : {total_tokens:,}")
    print(f"  Total Documents  : {total_docs:,}")
    print(f"  Total Shards     : {len(shards_manifest)}")
    print(f"  Time Elapsed     : {elapsed:.2f}s ({report['tokens_per_second']:.1f} tokens/s)")
    print(f"  Manifest Saved   : {manifest_path}")
    print("=" * 80)
    return report

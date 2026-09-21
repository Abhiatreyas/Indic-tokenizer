"""Tests for HuggingFace wrapper and parallel sharding pipeline."""

import os
import shutil
import pytest
from tokenizer.hf_tokenizer import IndicPartitionedTokenizer
from tokenizer.pipeline import shard_corpus

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "slot_demo")


@pytest.fixture
def hf_tok():
    if not os.path.exists(DEMO_DIR):
        pytest.skip("slot_demo artifact missing")
    return IndicPartitionedTokenizer.from_pretrained(DEMO_DIR)


def test_hf_basic_encode_decode(hf_tok):
    text = "Hello world! ನಮಸ್ಕಾರ!"
    ids = hf_tok.encode(text, add_special_tokens=False)
    assert len(ids) > 0
    decoded = hf_tok.decode(ids)
    assert decoded == text


def test_hf_special_tokens(hf_tok):
    text = "നമസ്കാരം"
    ids = hf_tok.encode(text, add_special_tokens=True)
    assert ids[0] == hf_tok.bos_token_id
    assert ids[-1] == hf_tok.eos_token_id
    decoded = hf_tok.decode(ids, skip_special_tokens=True)
    assert decoded == text


def test_hf_batch_call(hf_tok):
    batch = ["Hello world", "ನಮಸ್ಕಾರ"]
    res = hf_tok(batch, padding=True, return_tensors=None)
    assert "input_ids" in res
    assert "attention_mask" in res
    assert len(res["input_ids"]) == 2
    # Second sequence should have padding
    assert len(res["input_ids"][0]) == len(res["input_ids"][1])


def test_sharding_pipeline(tmp_path):
    sample_file = tmp_path / "sample.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        for i in range(50):
            f.write(f'{{"text": "Hello world document {i} ನಮಸ್ಕಾರ"}}\n')

    out_dir = str(tmp_path / "shards")
    report = shard_corpus(
        input_paths=[str(sample_file)],
        output_dir=out_dir,
        tokenizer_dir=DEMO_DIR,
        shard_size_tokens=200,
        num_workers=1,
    )
    assert report["total_docs"] == 50
    assert report["total_tokens"] > 0
    assert report["num_shards"] >= 1
    assert os.path.exists(os.path.join(out_dir, "shards_manifest.json"))

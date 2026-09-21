"""HuggingFace PreTrainedTokenizer wrapper for PartitionedTokenizer.

Provides complete drop-in compatibility with the HuggingFace Transformers ecosystem:
    - AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    - PyTorch BatchEncoding (input_ids, attention_mask as torch.Tensor)
    - save_pretrained() and from_pretrained()
    - Padding, truncation, special tokens handling (<bos>, <eos>, <pad>, <unk>)
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from transformers import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding
except ImportError:
    # Graceful fallback if transformers is not installed
    PreTrainedTokenizer = object
    BatchEncoding = dict

from .merge import (
    MODELS_DIRNAME,
    SLOTS_NAME,
    SPECS_NAME,
    VOCAB_NAME,
)
from .partitioned import PartitionedTokenizer


class IndicPartitionedTokenizer(PreTrainedTokenizer):
    """Production HuggingFace wrapper for Script-Partitioned Tokenizer."""

    vocab_files_names = {
        "vocab_file": VOCAB_NAME,
        "slots_file": SLOTS_NAME,
        "specs_file": SPECS_NAME,
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_dir: Optional[str] = None,
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask\>",
        **kwargs: Any,
    ) -> None:
        self.tokenizer_dir = tokenizer_dir
        if tokenizer_dir and os.path.exists(tokenizer_dir):
            self._impl: Optional[PartitionedTokenizer] = PartitionedTokenizer.from_dir(
                tokenizer_dir
            )
        else:
            self._impl = None

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._impl.layout.vocab_size if self._impl else 0

    def get_vocab(self) -> Dict[str, int]:
        if not self._impl:
            return {}
        return {tok: i for i, tok in enumerate(self._impl.vocab) if tok}

    def _tokenize(self, text: str, **kwargs: Any) -> List[str]:
        if not self._impl:
            return []
        ids = self._impl.encode(text)
        return [self._convert_id_to_token(i) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        if not self._impl:
            return 0
        try:
            return self._impl.vocab.index(token)
        except ValueError:
            return self.unk_token_id if self.unk_token_id is not None else 3

    def _convert_id_to_token(self, index: int) -> str:
        if not self._impl or index >= len(self._impl.vocab) or index < 0:
            return self.unk_token or "<unk>"
        return self._impl.vocab[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = [self._convert_token_to_id(t) for t in tokens]
        return self.decode(ids)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        if not self._impl:
            return []
        ids = self._impl.encode(text)
        if add_special_tokens:
            bos = [self.bos_token_id] if self.bos_token_id is not None else []
            eos = [self.eos_token_id] if self.eos_token_id is not None else []
            ids = bos + ids + eos
        return ids

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if not self._impl:
            return ""
        if skip_special_tokens:
            specials = set(range(self._impl.layout.special_end))
            token_ids = [i for i in token_ids if i not in specials]
        return self._impl.decode(token_ids)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        if not self.tokenizer_dir or not os.path.exists(self.tokenizer_dir):
            return ()

        for fname in (VOCAB_NAME, SLOTS_NAME, SPECS_NAME, "tokenizer_v1.manifest.json"):
            src = os.path.join(self.tokenizer_dir, fname)
            dst = os.path.join(save_directory, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        src_models = os.path.join(self.tokenizer_dir, MODELS_DIRNAME)
        dst_models = os.path.join(save_directory, MODELS_DIRNAME)
        if os.path.exists(src_models):
            if os.path.exists(dst_models):
                shutil.rmtree(dst_models)
            shutil.copytree(src_models, dst_models)

        # Write auto_map so HuggingFace hub automatically loads this class
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        cfg = {
            "tokenizer_class": "IndicPartitionedTokenizer",
            "auto_map": {
                "AutoTokenizer": [
                    "tokenizer.hf_tokenizer.IndicPartitionedTokenizer",
                    None,
                ]
            },
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
        }
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

        return (
            os.path.join(save_directory, VOCAB_NAME),
            os.path.join(save_directory, SLOTS_NAME),
            os.path.join(save_directory, SPECS_NAME),
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> "IndicPartitionedTokenizer":
        return cls(tokenizer_dir=pretrained_model_name_or_path, **kwargs)

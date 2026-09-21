"""Quantify the per-slot byte-alphabet tax.

Every trained slot seeds the full 256-character ByteLevel alphabet via
`initial_alphabet` so that it can represent any input losslessly on its own. That
costs 256 vocabulary ids PER SLOT, before a single learned subword.

Hypothesis: at a matched total vocabulary the partitioned design loses badly to a
shared-vocabulary baseline because most of its budget is spent on that tax.

Run after a build exists:
    python diagnostics/diagnose_byte_tax.py artifacts/ws_leading artifacts/ws_leading_small
"""
from __future__ import annotations

import json
import os
import sys

from tokenizers import pre_tokenizers

ALPHABET = set(pre_tokenizers.ByteLevel.alphabet())


def report(out_dir: str) -> None:
    with open(os.path.join(out_dir, "vocab.v1.json"), encoding="utf-8") as fh:
        vd = json.load(fh)
    with open(os.path.join(out_dir, "slots.v1.json"), encoding="utf-8") as fh:
        layout = json.load(fh)

    tokens = vd["tokens"]
    total = vd["vocab_size"]

    print(f"\n=== {out_dir}  vocab_size={total} "
          f"padded={vd.get('padded_vocab_size')} "
          f"whitespace={vd.get('whitespace_ownership')} ===")

    grand_bytes = 0
    for s in layout["slots"]:
        chunk = tokens[s["start"]:s["end"]]
        n_byte = sum(1 for t in chunk if t in ALPHABET)
        learned = s["size"] - n_byte
        grand_bytes += n_byte
        print(
            f"  {s['name']:<14} size={s['size']:<6} raw-byte={n_byte:<5} "
            f"learned={learned:<6} byte share={n_byte / s['size']:.1%}"
        )

    print(f"  {'TOTAL':<14} byte rows={grand_bytes} "
          f"({grand_bytes / total:.1%} of vocabulary), "
          f"learned rows={total - len(layout['special_tokens']) - grand_bytes}")


if __name__ == "__main__":
    for d in sys.argv[1:] or ["artifacts/ws_leading"]:
        report(d)

"""Show how slot separation actually works, mechanically.

Answers "is every row in the v3 table its own slot?": yes, but the slots are NOT
separate vocabularies or separate embedding tables. They are contiguous id RANGES
inside one merged vocabulary, and the range an id falls in is what tells the
decoder which tokenizer to use.

Builds the v3 slot structure (small vocabularies for speed -- the point is the
mechanism, not the sizes) and then walks a mixed-script sentence through it.

    python diagnostics/explain_slot_layout.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

from tokenizer.merge import freeze
from tokenizer.partitioned import PartitionedTokenizer
from tokenizer.scripts import CATCHALL, CODE_MATH, COMMON, LATIN, NEUTRAL
from tokenizer.slots import IdLayout, SlotSpec, slot_map_from_specs
from tokenizer.train import ByteFallbackSlot, train_slot

ML = "data_real/multilang"

# The v3 design: one slot per row. (name, algorithm, scripts, source paths)
# NEUTRAL and CODE_MATH get real corpora too -- an earlier version of this demo
# trained them on a single dummy run, which left whitespace unrepresentable and
# made spaces fall through to CATCHALL, misrepresenting the design.
V3_SLOTS = [
    ("NEUTRAL", "bpe", (COMMON,), "data_real/en_train.jsonl"),
    ("LATIN", "unigram", (LATIN,), "data_real/en_train.jsonl"),
    ("DEVANAGARI", "unigram", ("DEVANAGARI",), "data_real/hi_train.jsonl"),
    ("KANNADA", "unigram", ("KANNADA",), "data_real/kn_train.jsonl"),
    ("MALAYALAM", "unigram", ("MALAYALAM",), f"{ML}/ml.jsonl"),
    ("TELUGU", "unigram", ("TELUGU",), f"{ML}/te.jsonl"),
    ("TAMIL", "unigram", ("TAMIL",), f"{ML}/ta.jsonl"),
    ("BENGALI", "unigram", ("BENGALI",), f"{ML}/bn.jsonl"),
    ("GUJARATI", "unigram", ("GUJARATI",), f"{ML}/gu.jsonl"),
    ("ORIYA", "unigram", ("ORIYA",), f"{ML}/or.jsonl"),
    ("GURMUKHI", "unigram", ("GURMUKHI",), f"{ML}/pa.jsonl"),
    ("CODE_MATH", "bpe", (), "CODE"),
]

DEMO_VOCAB = 1500  # small, for speed; the real table uses 16k-64k
MAX_DOCS = 900


def load_docs(path: str, cap: int) -> List[str]:
    out, total = [], 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)["text"])
            total += len(line)
            if total >= cap or len(out) >= MAX_DOCS:
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/slot_demo")
    ap.add_argument("--vocab", type=int, default=DEMO_VOCAB)
    args = ap.parse_args(argv)

    specs = tuple(
        SlotSpec(name, algo, scripts, structural=(name == CODE_MATH),
                 max_piece_length=48)
        for name, algo, scripts, _ in V3_SLOTS
    ) + (SlotSpec(CATCHALL, "bytes", (), trained=False,
                  exempt_from_budget=True),)

    slot_map = slot_map_from_specs(specs)
    print("ROUTING RULE (script -> slot), computed from the specs:")
    for script, slot in sorted(slot_map.items()):
        print(f"    {script:<12} -> {slot}")
    print(f"    everything else -> {CATCHALL} (byte fallback)")

    # ---- train one tokenizer per slot -----------------------------------
    print(f"\nTRAINING (vocab {args.vocab} each, truncated corpora for speed)")
    slots: Dict[str, object] = {}
    for name, algo, scripts, path in V3_SLOTS:
        from tokenizer.scripts import CODE_MATH as CM, segment

        runs: List[str] = []
        if path == "CODE":
            # CODE_MATH is routed structurally, so real code is wrapped in fences
            # the way a markdown document would present it.
            src = "data_real/en_train.jsonl"
            if os.path.exists(src):
                for doc in load_docs(src, 200_000):
                    runs.append("```\n" + doc[:400] + "\n```")
        elif path and os.path.exists(path):
            for doc in load_docs(path, 600_000):
                for r in segment(doc, slot_map=slot_map, structural_slot=CM,
                                 whitespace_ownership="leading"):
                    if r.slot == name:
                        runs.append(r.text)
        if len(runs) < 50:
            raise SystemExit(
                f"slot {name!r} collected only {len(runs)} runs from {path!r}; "
                "the demo needs real data for every slot"
            )
        slots[name] = train_slot(f"{name}_demo", algo, runs, args.vocab,
                                 max_piece_length=48)
        print(f"    {name:<12} {algo:<8} trained on {len(runs):>6} runs "
              f"-> {slots[name].local_size} pieces")
    slots[CATCHALL] = ByteFallbackSlot()

    # ---- merge into ONE vocabulary --------------------------------------
    specials = ("<pad>", "<bos>", "<eos>", "<unk>", "<mask>")
    manifest = freeze(args.out, specials, slots, specs)
    tok = PartitionedTokenizer.from_dir(args.out)

    print(f"\nONE MERGED VOCABULARY: {manifest['vocab_size']} rows total")
    print(f"  {'id range':<20}{'slot':<12}{'algorithm':<10}{'rows':>7}")
    for s in tok.layout.slots:
        print(f"  {f'{s.start}..{s.end - 1}':<20}{s.name:<12}"
              f"{s.algorithm:<10}{s.size:>7}")
    print(f"  {'specials':<20}{'':<12}{'':<10}{len(specials):>7}")

    # ---- walk a mixed-script sentence through ---------------------------
    text = "hello नमस्ते ಕನ್ನಡ தமிழ், def f(x): return x"
    print(f"\nROUTING A MIXED SENTENCE\n  input: {text!r}\n")
    print(f"  {'run':<18}{'script':<12}{'-> slot':<12}{'ids (global)':<26}")
    for run, ids in tok.encode_runs(text):
        rng = tok.lookup.slot_of(ids[0]) if ids else None
        head = ids[:6]
        tail = "..." if len(ids) > 6 else ""
        print(f"  {run.text[:16]!r:<18}{run.script:<12}{rng.name:<12}"
              f"{str(head) + tail:<26}")
    ids = tok.encode(text)
    print(f"\n  full id stream: {ids}")
    print(f"  note: ranges are interleaved in original text order, not grouped")
    print(f"\n  decoded back: {tok.decode(ids)!r}")
    print(f"  round-trip exact: {tok.decode(ids) == text}")

    print("\nDECODING IS A RANGE LOOKUP, NOT LANGUAGE DETECTION:")
    for probe in ids[:10]:
        rng = tok.lookup.slot_of(probe)
        label = rng.name if rng else "special"
        piece = tok.vocab[probe]
        print(f"    id {probe:<6} -> range {rng.start}..{rng.end - 1} "
              f"-> {label:<12} -> token {piece[:24]!r}"
              if rng else f"    id {probe:<6} -> special token {piece!r}")

    print("\nCATCHALL: a slot with NO trained vocabulary, just 256 byte rows.")
    print("It catches scripts that have no slot -- one token per UTF-8 byte:")
    for s in ("العربية", "中文", "සිංහල", "ქართული"):
        sub = tok.encode(s)
        names = {tok.slot_of_id(i) for i in sub}
        print(f"    {s!r:<12} {len(sub):>3} ids  slots={names}  "
              f"UTF-8 bytes={len(s.encode('utf-8'))}")
    print("    (Tamil and Kannada have their own slots in v3, so they do NOT "
          "fall here -- see the layout above.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

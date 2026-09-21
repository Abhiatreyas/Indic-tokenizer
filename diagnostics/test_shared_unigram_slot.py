"""One shared Unigram slot vs per-script slots, at equal total budget.

The proposal: since there are only two algorithms, have only two slots -- one big
Unigram slot with a shared vocabulary, plus BPE for the rest. This tests whether a
single vocabulary spanning many SCRIPTS can match one vocabulary per script.

This is not the same as section 14's shared-Devanagari result, where three
languages shared one script's byte space. Here Tamil, Bengali, Gurmukhi and
Devanagari all compete for the same rows, and their UTF-8 byte ranges barely
overlap, so most rows only ever fire for one script. sarvam-1 is the closest
real-world instance: 68k shared across 10 Indic languages, 30% worse on Kannada
than a dedicated 32k slot, at 24.2% utilization.

Three configurations at the SAME total vocabulary:

  A per-script   -- one Unigram slot per script (the current design)
  B one-shared   -- a single Unigram slot holding every script
  C indic+latin  -- one shared Indic slot, LATIN separate

Per-script budgets follow the SPEC v2 proportions, scaled to the chosen total.

    python diagnostics/test_shared_unigram_slot.py --total 128000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

from tokenizer.train import (
    ALPHABET_OBSERVED_ASCII,
    UnrepresentableRun,
    train_slot,
)

ML = "data_real/multilang"
RO = "data_real/romanized"

# script -> (train paths, held path)
SCRIPTS: Dict[str, Tuple[List[str], str]] = {
    "LATIN": (["data_real/en_train.jsonl", f"{RO}/hinglish.jsonl",
               f"{RO}/tanglish.jsonl"], "data_real/en_heldout.jsonl"),
    "DEVANAGARI": (["data_real/hi_train.jsonl", f"{ML}/mr.jsonl"],
                   "data_real/hi_heldout.jsonl"),
    "KANNADA": (["data_real/kn_train.jsonl"], "data_real/kn_heldout.jsonl"),
    "MALAYALAM": ([f"{ML}/ml.jsonl"], f"{ML}/ml.jsonl"),
    "TELUGU": ([f"{ML}/te.jsonl"], f"{ML}/te.jsonl"),
    "TAMIL": ([f"{ML}/ta.jsonl"], f"{ML}/ta.jsonl"),
    "BENGALI": ([f"{ML}/bn.jsonl"], f"{ML}/bn.jsonl"),
    "GUJARATI": ([f"{ML}/gu.jsonl"], f"{ML}/gu.jsonl"),
    "ORIYA": ([f"{ML}/or.jsonl"], f"{ML}/or.jsonl"),
    "GURMUKHI": ([f"{ML}/pa.jsonl"], f"{ML}/pa.jsonl"),
}

# SPEC v2 shares, scaled to the total. Shared slots get more, easy scripts less.
V2_SHARES = {
    "LATIN": 64000, "DEVANAGARI": 48000, "KANNADA": 48000, "MALAYALAM": 48000,
    "TELUGU": 48000, "TAMIL": 40000, "BENGALI": 32000, "GUJARATI": 24000,
    "ORIYA": 24000, "GURMUKHI": 20000,
}


def load_docs(path: str, cap: int) -> List[str]:
    out, total = [], 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)["text"]
            out.append(t)
            total += len(t)
            if total >= cap:
                break
    return out


def runs_for(docs: Sequence[str], script: str) -> List[str]:
    from tokenizer.scripts import CODE_MATH, COMMON, LATIN, NEUTRAL, segment

    slot_map = {COMMON: NEUTRAL, LATIN: LATIN, script: script}
    out = []
    for t in docs:
        for r in segment(t, slot_map=slot_map, structural_slot=CODE_MATH,
                         whitespace_ownership="leading"):
            if r.slot == script:
                out.append(r.text)
    return out


def count(slot, runs: Sequence[str]) -> Tuple[int, int]:
    """(tokens, distinct piece ids fired)."""
    total, fired = 0, set()
    for text in runs:
        stack = [text]
        while stack:
            c = stack.pop()
            if not c:
                continue
            try:
                ids = slot.encode(c)
                total += len(ids)
                fired.update(ids)
                continue
            except UnrepresentableRun:
                pass
            if len(c) == 1:
                bs = list(c.encode("utf-8"))
                total += len(bs)
                fired.update(bs)
                continue
            mid = len(c) // 2
            stack.append(c[mid:])
            stack.append(c[:mid])
    return total, len(fired)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=128_000)
    ap.add_argument("--train-chars", type=int, default=500_000,
                    help="per script (Latin splits it across 3 sources)")
    ap.add_argument("--held-chars", type=int, default=150_000)
    ap.add_argument("--out", default="artifacts/shared_slot_test.json")
    args = ap.parse_args(argv)

    # ---- prepare runs ---------------------------------------------------
    train_runs: Dict[str, List[str]] = {}
    held_runs: Dict[str, List[str]] = {}
    for script, (paths, held) in SCRIPTS.items():
        per = max(1, args.train_chars // max(1, len(paths)))
        tr: List[str] = []
        for p in paths:
            if os.path.exists(p):
                tr += runs_for(load_docs(p, per), script)
        he = runs_for(load_docs(held, args.held_chars), script) \
            if os.path.exists(held) else []
        if len(tr) < 100 or len(he) < 50:
            print(f"  {script}: insufficient data, skipping")
            continue
        train_runs[script] = tr
        held_runs[script] = he
        print(f"  {script:<12} train_runs={len(tr):>7} "
              f"({sum(len(r) for r in tr):>9} ch)  held_words="
              f"{sum(len(r.split()) for r in he):>7}")

    scripts = list(train_runs)
    all_train = [r for s in scripts for r in train_runs[s]]
    print(f"\n  pooled across {len(scripts)} scripts: {len(all_train)} runs "
          f"({sum(len(r) for r in all_train)} chars)")

    # per-script budgets summing to total
    share_sum = sum(V2_SHARES[s] for s in scripts)
    budgets = {s: max(2000, round(args.total * V2_SHARES[s] / share_sum))
               for s in scripts}
    print(f"  per-script budgets sum={sum(budgets.values())} "
          f"(target {args.total})")

    rows: List[dict] = []

    def evaluate(cfg_name: str, slot_for: Dict[str, object]):
        for s in scripts:
            words = sum(len(r.split()) for r in held_runs[s])
            toks, fired = count(slot_for[s], held_runs[s])
            rows.append({"config": cfg_name, "script": s,
                         "tokens_per_word": round(toks / words, 4),
                         "pieces": getattr(slot_for[s], "local_size", None),
                         "fired": fired})
            print(f"     {cfg_name:<14} {s:<12} tok/word={toks/words:.4f}")

    # ---- A: per-script slots -------------------------------------------
    print(f"\n[A] per-script slots, each its own vocabulary")
    t0 = time.time()
    slotA = {}
    for s in scripts:
        slotA[s] = train_slot(s, "unigram", list(train_runs[s]), budgets[s],
                              alphabet_scope=ALPHABET_OBSERVED_ASCII,
                              max_piece_length=48)
    print(f"   trained in {time.time()-t0:.0f}s")
    evaluate("A_per_script", slotA)

    # ---- B: one shared slot for everything -----------------------------
    print(f"\n[B] ONE shared Unigram slot across all {len(scripts)} scripts, "
          f"vocab {args.total}")
    t0 = time.time()
    shared = train_slot("SHARED", "unigram", all_train, args.total,
                        alphabet_scope=ALPHABET_OBSERVED_ASCII,
                        max_piece_length=48)
    print(f"   trained in {time.time()-t0:.0f}s, pieces={shared.local_size}")
    evaluate("B_one_shared", {s: shared for s in scripts})

    # ---- C: shared Indic slot, LATIN separate --------------------------
    print(f"\n[C] one shared INDIC slot + separate LATIN")
    indic = [s for s in scripts if s != "LATIN"]
    indic_train = [r for s in indic for r in train_runs[s]]
    lt = time.time()
    indic_slot = train_slot("INDIC", "unigram", indic_train,
                            budgets.get("LATIN", 0) and
                            (sum(budgets[s] for s in indic)),
                            alphabet_scope=ALPHABET_OBSERVED_ASCII,
                            max_piece_length=48)
    latin_slot = train_slot("LATIN", "unigram", list(train_runs["LATIN"]),
                            budgets["LATIN"],
                            alphabet_scope=ALPHABET_OBSERVED_ASCII,
                            max_piece_length=48)
    print(f"   trained in {time.time()-lt:.0f}s, "
          f"indic pieces={indic_slot.local_size}, "
          f"latin pieces={latin_slot.local_size}")
    evaluate("C_indic_shared",
             {s: (latin_slot if s == "LATIN" else indic_slot) for s in scripts})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== held-out tokens/word, same total vocabulary "
          f"({args.total}) ===")
    print(f"{'script':<12}{'A per-script':>14}{'B one-shared':>14}"
          f"{'C indic+latin':>15}{'B vs A':>9}{'C vs A':>9}")
    by = {}
    for r in rows:
        by.setdefault(r["script"], {})[r["config"]] = r["tokens_per_word"]
    tot = {"A": 0.0, "B": 0.0, "C": 0.0}
    for s in scripts:
        d = by.get(s, {})
        a = d.get("A_per_script", float("nan"))
        b = d.get("B_one_shared", float("nan"))
        c = d.get("C_indic_shared", float("nan"))
        tot["A"] += a; tot["B"] += b; tot["C"] += c
        print(f"{s:<12}{a:>14.4f}{b:>14.4f}{c:>15.4f}"
              f"{(b-a)/a*100:>8.1f}%{(c-a)/a*100:>8.1f}%")
    n = len(scripts)
    print(f"{'MEAN':<12}{tot['A']/n:>14.4f}{tot['B']/n:>14.4f}"
          f"{tot['C']/n:>15.4f}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

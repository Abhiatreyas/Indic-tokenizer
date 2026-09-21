"""Fetch text corpora for romanized code-mixed Indic ("Hinglish", "Tanglish").

There is no public Kanglish corpus on the HuggingFace Hub (searched: kanglish,
kannada romanized, kannada english mixed, kannada transliteration -- all empty).
Tanglish, the Tamil equivalent, does exist and is the closest available proxy for
romanized Dravidian+English, so it stands in for Kanglish and that substitution is
recorded rather than hidden.

Schemas are discovered from the datasets-server rather than assumed.

    python diagnostics/fetch_romanized_mix.py --target-chars 3000000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co"
PAGE = 100
SLEEP = 0.6
MAX_RETRIES = 6

# label -> (dataset, config, split, note)
SOURCES = {
    "hinglish": (
        "Abhishekcr448/Hinglish-Everyday-Conversations-1M", "default", "train",
        "real Hinglish (romanized Hindi + English)",
    ),
    "hinglish_news": (
        "suyash2739/News_Hinglish_English", "default", "train",
        "real Hinglish, news register",
    ),
    "tanglish": (
        "vishnu-n/Tanglish-Corpus-185k", "default", "train",
        "real Tanglish (romanized Tamil + English) -- proxy for Kanglish",
    ),
}


def _get(url: str):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "tokenizer-research/0.1"}
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                time.sleep(2 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"giving up: {last}")


def discover_string_fields(dataset: str, config: str, split: str) -> list:
    url = (f"{API}/first-rows?dataset={urllib.parse.quote(dataset)}"
           f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}")
    d = _get(url)
    feats = (d.get("features") or [])
    out = []
    for f in feats:
        if f.get("type", {}).get("dtype") == "string":
            out.append(f["name"])
    return out


def fetch(label: str, target_chars: int, out_dir: str) -> dict | None:
    dataset, config, split, note = SOURCES[label]
    path = os.path.join(out_dir, f"{label}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        n = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {label}: already have {n} docs, skipping")
        return {"label": label, "dataset": dataset, "docs": n,
                "bytes": os.path.getsize(path), "note": note, "cached": True}

    fields = discover_string_fields(dataset, config, split)
    print(f"  {label}: dataset={dataset} string_fields={fields}")
    if not fields:
        print(f"  {label}: no string fields, skipping")
        return None

    total, docs, offset = 0, 0, 0
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        while total < target_chars:
            url = (f"{API}/rows?dataset={urllib.parse.quote(dataset)}"
                   f"&config={urllib.parse.quote(config)}"
                   f"&split={urllib.parse.quote(split)}&offset={offset}&length={PAGE}")
            try:
                rows = _get(url).get("rows", [])
            except RuntimeError as e:
                print(f"      {label}: {e}; stopping with {total} chars")
                break
            if not rows:
                break
            for item in rows:
                row = item.get("row", {})
                parts = [str(row[f]).strip() for f in fields
                         if isinstance(row.get(f), str) and row[f].strip()]
                if not parts:
                    continue
                text = " ".join(parts)
                if len(text) < 20:
                    continue
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                total += len(text)
                docs += 1
            offset += PAGE
            print(f"      {label}: offset={offset} docs={docs} chars={total}",
                  flush=True)
            time.sleep(SLEEP)
    os.replace(tmp, path)
    return {"label": label, "dataset": dataset, "docs": docs, "chars": total,
            "note": note, "cached": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data_real/romanized")
    ap.add_argument("--target-chars", type=int, default=3_000_000)
    ap.add_argument("--labels", default="hinglish,tanglish")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for label in [s.strip() for s in args.labels.split(",") if s.strip()]:
        if label not in SOURCES:
            print(f"  unknown label {label!r}")
            continue
        print(f"[fetch] {label}: {SOURCES[label][3]}")
        try:
            info = fetch(label, args.target_chars, args.out)
            if info:
                summary.append(info)
        except Exception as e:  # noqa: BLE001
            print(f"  {label} FAILED: {type(e).__name__}: {e}")

    print()
    for s in summary:
        print(f"  {s['label']:<16} docs={s['docs']:>7} chars={s.get('chars', '?')}")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwritten: {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

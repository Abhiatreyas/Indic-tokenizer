"""Fetch small per-language Indic samples from Wikipedia for tokenizer benchmarking.

The local Indic corpora turned out to be Kannada-only (both `indiccorp_v2.jsonl`
and `sangraha_verified.jsonl` start with, and stay on, `group=kannada`), so other
scripts need another source. Wikipedia via the HF datasets-server is labelled,
clean, and covers every major Indic language.

Each language gets one block of text which is later split article-level into
train/held-out, so this makes ~2 requests per language. The API rate-limits, so
requests are spaced and retried with backoff. Already-fetched languages are
skipped, making the run resumable.

    python diagnostics/fetch_indic_multilang.py --target-chars 1800000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://datasets-server.huggingface.co/rows"

# language code -> (wikipedia config, expected Unicode script, English name)
LANGS = {
    "hi": ("20231101.hi", "DEVANAGARI", "Hindi"),
    "mr": ("20231101.mr", "DEVANAGARI", "Marathi"),
    "ne": ("20231101.ne", "DEVANAGARI", "Nepali"),
    "bn": ("20231101.bn", "BENGALI", "Bengali"),
    "ta": ("20231101.ta", "TAMIL", "Tamil"),
    "te": ("20231101.te", "TELUGU", "Telugu"),
    "ml": ("20231101.ml", "MALAYALAM", "Malayalam"),
    "gu": ("20231101.gu", "GUJARATI", "Gujarati"),
    "pa": ("20231101.pa", "GURMUKHI", "Punjabi"),
    "or": ("20231101.or", "ORIYA", "Odia"),
    "ur": ("20231101.ur", "ARABIC", "Urdu"),
}

PAGE = 100
MAX_ARTICLE_CHARS = 20_000
SLEEP_BETWEEN = 2.0
MAX_RETRIES = 6


def fetch_page(config: str, offset: int) -> list:
    url = (
        f"{API}?dataset=wikimedia%2Fwikipedia&config={config}"
        f"&split=train&offset={offset}&length={PAGE}"
    )
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "tokenizer-research/0.1"}
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read()).get("rows", [])
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"      429, backing off {wait}s", flush=True)
                time.sleep(wait)
            else:
                time.sleep(2 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"giving up after {MAX_RETRIES}: {last}")


def fetch_lang(code: str, target_chars: int, out_dir: str) -> dict | None:
    config, script, name = LANGS[code]
    path = os.path.join(out_dir, f"{code}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        n = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {code} ({name}): already have {n} docs, skipping")
        return {"lang": code, "script": script, "name": name, "path": path,
                "docs": n, "chars": os.path.getsize(path), "cached": True}

    total, docs, offset = 0, 0, 0
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        while total < target_chars:
            try:
                rows = fetch_page(config, offset)
            except RuntimeError as e:
                print(f"      {code}: {e}; keeping {total} chars fetched so far")
                break
            if not rows:
                break
            for item in rows:
                row = item.get("row", {})
                text = (row.get("text") or "").strip()
                if len(text) < 100:
                    continue
                text = text[:MAX_ARTICLE_CHARS]
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                total += len(text)
                docs += 1
            offset += PAGE
            print(f"      {code}: offset={offset} docs={docs} chars={total}",
                  flush=True)
            time.sleep(SLEEP_BETWEEN)
    os.replace(tmp, path)
    return {"lang": code, "script": script, "name": name, "path": path,
            "docs": docs, "chars": total, "cached": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data_real/multilang")
    ap.add_argument("--target-chars", type=int, default=1_800_000)
    ap.add_argument("--langs", default=",".join(LANGS))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for code in [c.strip() for c in args.langs.split(",") if c.strip()]:
        if code not in LANGS:
            print(f"  unknown lang {code!r}, skipping")
            continue
        print(f"[fetch] {code} ({LANGS[code][2]}) target={args.target_chars}")
        try:
            info = fetch_lang(code, args.target_chars, args.out)
            if info:
                summary.append(info)
        except Exception as e:  # noqa: BLE001
            print(f"  {code} FAILED: {type(e).__name__}: {e}")

    print()
    print(f"  {'lang':<5}{'script':<14}{'docs':>8}{'chars':>12}")
    for s in summary:
        print(f"  {s['lang']:<5}{s['script']:<14}{s['docs']:>8}{s['chars']:>12}")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwritten: {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

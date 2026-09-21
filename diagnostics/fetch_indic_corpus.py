"""Fetch a real Indic corpus from the HuggingFace datasets-server.

The synthetic generator has a fixed word bank: ~101 distinct Hindi word forms and
75 distinct Devanagari runs. That is close to a worst case for Unigram, whose EM
needs lexical diversity, and scaling the generator up adds combinations rather
than new words. So testing "Unigram vs BPE for Indic" needs real text.

Source: wikimedia/wikipedia via the datasets-server rows API. Fetched content is
data only -- it is written to disk as corpus text and never interpreted.

    python diagnostics/fetch_indic_corpus.py --target-chars 4000000
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
CONFIGS = {
    "hi": "20231101.hi",  # Hindi (Devanagari)
    "kn": "20231101.kn",  # Kannada
}
PAGE = 100
MAX_ARTICLE_CHARS = 20_000  # keep one huge article from dominating


def fetch_page(config: str, offset: int, retries: int = 4) -> list:
    url = (
        f"{API}?dataset=wikimedia%2Fwikipedia&config={config}"
        f"&split=train&offset={offset}&length={PAGE}"
    )
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "tokenizer-research/0.1"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read()).get("rows", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"giving up after {retries} attempts: {last}")


def fetch_language(
    lang: str,
    target_chars: int,
    out_dir: str,
    out_name: str | None = None,
    start_offset: int = 0,
) -> dict:
    config = CONFIGS[lang]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_name or f"wiki_{lang}.jsonl")

    total = 0
    articles = 0
    offset = start_offset
    with open(path, "w", encoding="utf-8") as fh:
        while total < target_chars:
            rows = fetch_page(config, offset)
            if not rows:
                break
            for item in rows:
                row = item.get("row", {})
                text = (row.get("text") or "").strip()
                if len(text) < 50:
                    continue
                text = text[:MAX_ARTICLE_CHARS]
                fh.write(
                    json.dumps(
                        {"id": row.get("id"), "title": row.get("title"), "text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total += len(text)
                articles += 1
            offset += PAGE
            print(
                f"    {lang}: offset={offset} articles={articles} chars={total}",
                flush=True,
            )
            time.sleep(0.4)

    return {"lang": lang, "config": config, "path": path,
            "articles": articles, "chars": total, "start_offset": start_offset}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-chars", type=int, default=4_000_000)
    ap.add_argument("--out", default="data_real")
    ap.add_argument("--langs", default="hi,kn")
    ap.add_argument("--out-name", default=None,
                    help="output filename; use for disjoint train/held-out splits")
    ap.add_argument("--start-offset", type=int, default=0,
                    help="row offset to start from; different offsets give disjoint sets")
    args = ap.parse_args(argv)

    summary = []
    for lang in [s for s in args.langs.split(",") if s.strip()]:
        print(f"[fetch] {lang} -> target {args.target_chars} chars "
              f"from offset {args.start_offset}")
        summary.append(
            fetch_language(
                lang, args.target_chars, args.out,
                out_name=args.out_name, start_offset=args.start_offset,
            )
        )

    print()
    for s in summary:
        print(
            f"  {s['lang']}: {s['articles']} articles, {s['chars']} chars -> {s['path']}"
        )
    with open(os.path.join(args.out, "fetch_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

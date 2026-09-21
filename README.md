# Indic-Partitioned Tokenizer (`indic-tokenizer`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-180%2F180%20passing-brightgreen.svg)](tests/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Compatible-orange)](https://huggingface.co/)

A high-performance, script-partitioned multilingual tokenizer engineered to eliminate **"Script Starvation"** in foundation Large Language Models (LLMs).

---

## 1. The Core Problem: "Script Starvation" in Modern LLMs

Standard tokenizers (such as Llama-3, Gemma-2, or Sarvam) pool all languages into a single global vocabulary dictionary:
* **English & Code Greedily Dominate**: Because 85%+ of internet training data is English and code, frequency-greedy algorithms (BPE) give English millions of multi-word tokens, while South Indic languages get starved.
* **Extreme Fertility Penalty**: In standard tokenizers, Kannada, Telugu, Tamil, and Malayalam take **18 to 27 tokens per word**.
* **Byte Fallback Bloat**: Words trigger dozens of raw UTF-8 byte fallbacks (`<0xE0><0xB0><0xA8>...`).
* **Context Collapse**: A 4,096-token context window is consumed by just ~150 to 200 words of South Indic text, wasting compute and destroying long-context retrieval.

---

## 2. The Solution: Script-Partitioned Token ID Space

Instead of forcing all languages to fight in one giant pool, `indic-tokenizer` partitions the vocabulary into **independent, non-overlapping script slots**:

```
 0 .. k-1    [ SPECIAL TOKENS: <pad>, <bos>, <eos>, <unk>, <mask\> ]
 k .. N1     [ NEUTRAL: ASCII digits, whitespace, punctuation, emoji ] (BPE)
N1 .. N2     [ CODE_MATH: Python, C++, LaTeX syntax & indentation ]   (BPE)
N2 .. N3     [ LATIN: English, Kanglish, Hinglish, Tanglish ]          (Unigram)
N3 .. N4     [ DEVANAGARI: Hindi, Marathi, Nepali, Sanskrit ]          (Unigram)
N4 .. N5     [ KANNADA: Kannada ]                                      (Unigram)
N5 .. N6     [ TELUGU: Telugu ]                                        (Unigram)
N6 .. N7     [ TAMIL: Tamil ]                                          (Unigram)
N7 .. N8     [ MALAYALAM: Malayalam ]                                  (Unigram)
N8 .. N9     [ BENGALI, GUJARATI, GURMUKHI, ORIYA ]                    (Unigram)
N_last       [ CATCHALL: 256-Entry Byte Table ]                        (Lossless Fallback)
```

### What This Project Does (A Small Example)

Imagine you input a mixed sentence containing English, Python code, Kannada, and Hindi:

```text
print("ನಮಸ್ಕಾರ! नमस्ते!")
```

Here is step-by-step how `indic-tokenizer` processes it:

```
                            Input: print("ನಮಸ್ಕಾರ! नमस्ते!")
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                [1. Unicode Router]                  [2. Structural Parser]
             Identifies script blocks                Detects code syntax:
             by deterministic Unicode:               print(" ... ")
             • Kannada: 'ನಮಸ್ಕಾರ'
             • Devanagari: 'नमस्ते'
             • Punctuation: '!'
                        │                                     │
          ┌─────────────┼───────────────┬─────────────────────┘
          ▼             ▼               ▼
     ┌──────────┐  ┌──────────┐   ┌───────────┐
     │ KANNADA  │  │DEVANAGARI│   │ CODE/MATH │
     │ (Unigram)│  │ (Unigram)│   │   (BPE)   │
     └────┬─────┘  └────┬─────┘   └─────┬─────┘
          │             │               │
          ▼             ▼               ▼
     IDs: 4502..6000 IDs: 3003..4501 IDs: 16494..17993
          │             │               │
          └─────────────┼───────────────┘
                        ▼
           Final Global Token Stream:
       [16500, 16512, 4509, 5, 3009, 5, 16515]
```

1. **Deterministic Script Sorting**: Characters are routed strictly by Unicode properties into their respective departments (`KANNADA`, `DEVANAGARI`, `CODE_MATH`). There is no guessing and no error-prone machine-learned language-ID model.
2. **Hybrid Algorithms**:
   * **BPE** is assigned to `NEUTRAL` and `CODE_MATH` because programming syntax and indentation are rigid.
   * **Unigram (`max_piece_length=48`, `leading` whitespace)** is assigned to natural language scripts (`KANNADA`, `DEVANAGARI`, `LATIN`, `TELUGU`). Because Indian languages are agglutinative (rich in prefixes and suffixes), Unigram outperforms BPE by **5.5% to 9.7% lower fertility**.
3. **Guaranteed Allocation**: English and Hindi can never "steal" IDs from Kannada. Kannada words only compete with Kannada words.
4. **Decoding is Pure Math**: When the LLM emits ID `4509`, the decoder checks `4509 in [4502, 6000)` $\implies$ this is guaranteed Kannada. Local ID = $4509 - 4502 = 7 \implies$ `"ನ"`.
5. **100% Lossless Exact Invariant**: If a rare emoji 🦄 or unmodelled character appears, it falls back to the `CATCHALL` 256-byte table. It **never turns into `<unk>`**, and decodes back with 100% exact character fidelity.

---

## 3. How Does This Compare to Other Indic Tokenizers?

| Feature / Model | **`indic-partitioned-tokenizer`** | **Sarvam-1 (68K)** | **Llama-3 (128K)** | **Gemma-2 (256K)** |
| :--- | :---: | :---: | :---: | :---: |
| **Architecture** | **Script-Partitioned Slots** | Shared Global BPE | Shared Global BPE | Shared Global BPE |
| **Algorithm Strategy** | **Hybrid** (Unigram for Indic, BPE for Code) | BPE for all | BPE for all | BPE for all |
| **Kannada Fertility** | **1.73 tokens / word** | 2.48 tokens / word | ~3.80 tokens / word | ~2.20 tokens / word |
| **Tokens for 1,000 Kannada Words** | **1,730 tokens** (Most compact) | 2,480 tokens (+43% more) | 3,800 tokens (+120% more!) | 2,200 tokens (+27% more) |
| **Sequence Length Bloat vs Ours** | **1.0x (Baseline)** | 1.43x longer sequence | **2.20x longer sequence** | 1.27x longer sequence |
| **Vocabulary Utilization** | **79.7% active firing** | 24.2% active firing | < 20% on Indic | < 25% on Indic |
| **Total Vocab Size** | **128,000** | 68,000 | 128,000 | 256,000 (Bloated) |
| **Embedding Param Cost (8B Model)** | **~524M params** | ~278M params | ~524M params | ~1,048M params (2x cost!) |
| **Zero Unknown (`<unk>`)** | **100% Guaranteed (CATCHALL bytes)** | No | No | No |
| **Multi-Language Shared Robustness** | **Decisive Win (EM pooling)** | Diluted by English | Diluted by English | Diluted by English |

> **The Sarvam-1 Empirical Comparison**:
> In our diagnostics (`diagnostics/README.md` Section 17), Sarvam-1's 68k shared vocabulary scored **30.6% worse on Kannada** than a dedicated 32k Unigram slot (fertility 2.483 vs 1.735), with **3x worse vocabulary utilization** (24.2% vs 79.7%). Pooling multiple languages into a single shared BPE vocabulary forces scripts into destructive competition.

---

## 4. Empirical Benchmarks: Production vs. Demo Scale

### A. Production-Scale Benchmark (Target: 128K Total Vocabulary)
Measured on natural held-out evaluation text across real corpora (`diagnostics/README.md` Sections 14 & 16):

| Language | Script | Subwords Allocated | **Production Fertility (Tokens/Word)** | Compression (Bytes/Token) |
| :--- | :--- | :---: | :---: | :---: |
| **Kannada** | Kannada | 24,000 - 32,000 | **1.60 - 1.75** | ~4.3 B/tok |
| **Hindi** | Devanagari | 24,000 - 32,000 | **1.22 - 1.30** | ~3.3 B/tok |
| **Marathi** | Devanagari | (Shared Devanagari) | **1.70 - 1.84** | ~3.4 B/tok |
| **Telugu** | Telugu | 16,000 - 32,000 | **1.38 - 1.51** | ~4.3 B/tok |
| **Tamil** | Tamil | 16,000 - 32,000 | **1.24 - 1.34** | ~4.8 B/tok |
| **Malayalam** | Malayalam | 16,000 - 32,000 | **1.47 - 1.63** | ~4.8 B/tok |
| **Bengali** | Bengali | 16,000 - 32,000 | **1.01 - 1.18** | ~3.6 B/tok |
| **Gujarati** | Gujarati | 16,000 - 24,000 | **1.03 - 1.21** | ~3.5 B/tok |
| **Punjabi** | Gurmukhi | 16,000 - 20,000 | **0.86 - 1.01** | ~3.3 B/tok |
| **Odia** | Oriya | 16,000 - 24,000 | **1.00 - 1.14** | ~3.9 B/tok |
| **English** | Latin | 32,000 - 48,000 | **1.28 - 1.47** | ~1.8 B/tok |
| **Hinglish** | Latin (Romanized) | (Shared Latin) | **0.93 - 1.16** | ~1.4 B/tok |
| **Kanglish / Tanglish** | Latin (Romanized) | (Shared Latin) | **1.64 - 1.74** | ~1.5 B/tok |

---

### B. Minimal Smoke-Test Benchmark (`artifacts/slot_demo`)
*Why did the demo artifact test score 4.2–5.6 tok/word?*  
In `slot_demo`, each language was constrained to **only 1,499 tokens** (total 18K vocab). With only 1,499 subwords, words like `"ನಮಸ್ಕಾರ"` must be spelled out letter-by-letter (`ನ` + `ಮ` + `ಸ್` + `ಕಾರ` = 4 tokens). This confirms that slot boundaries work losslessly even under extreme constraints:

| Language | Script | Demo Vocab | **Demo Fertility** | Exact Round-Trip | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kannada** | Kannada | 1,499 | 5.17 tok/word | **PASS (100%)** | 328k tok/s |
| **Hindi** | Devanagari | 1,499 | 4.26 tok/word | **PASS (100%)** | 338k tok/s |
| **Telugu** | Telugu | 1,499 | 5.02 tok/word | **PASS (100%)** | 322k tok/s |
| **Tamil** | Tamil | 1,499 | 5.28 tok/word | **PASS (100%)** | 302k tok/s |
| **English** | Latin | 1,499 | 3.44 tok/word | **PASS (100%)** | 320k tok/s |
| **Hinglish** | Latin | 1,499 | 3.83 tok/word | **PASS (100%)** | 386k tok/s |

---

## 5. Code Setup & Installation

### Option A: Install via Pip (Editable)
```bash
git clone https://github.com/your-username/indic-partitioned-tokenizer.git
cd indic-partitioned-tokenizer
pip install -e .
```

### Option B: Install with Hugging Face & PyTorch Dependencies
```bash
pip install -e ".[hf]"
```

---

## 6. How to Run: Complete Workflow

### 1. Python API: Standard Usage
```python
from tokenizer import PartitionedTokenizer

# Load any built artifact
tok = PartitionedTokenizer.from_dir("artifacts/slot_demo")

# Encode text
text = "ನಮಸ್ಕಾರ! Hello world! नमस्ते!"
ids = tok.encode(text)
print("Token IDs:", ids)

# Decode back
decoded = tok.decode(ids)
assert decoded == text  # 100% exact character-for-character round-trip
```

---

### 2. HuggingFace Transformers & PyTorch Integration
```python
from tokenizer import IndicPartitionedTokenizer

# Load as a drop-in HuggingFace PreTrainedTokenizer
tokenizer = IndicPartitionedTokenizer.from_pretrained("artifacts/slot_demo")

# Encode with PyTorch tensors, padding, and attention masks
batch = ["Hello world!", "ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೆ"]
inputs = tokenizer(batch, padding=True, return_tensors="pt")

print("input_ids shape:", inputs["input_ids"].shape)
print("attention_mask:", inputs["attention_mask"])
```

---

### 3. Pre-Training Dataset Sharder (Multi-Processed)
Packs large JSONL corpora into contiguous binary `.bin` files (`uint16` / `uint32`) for PyTorch / LitGPT / NanoGPT pre-training:

```bash
indic-tokenizer shard \
    --inputs "data/corpus/kannada_clean.jsonl" "data/corpus/hindi_clean.jsonl" \
    --output "data/binary_shards" \
    --tokenizer-dir "artifacts/slot_demo" \
    --shard-size 50000000 \
    --workers 16
```
*Throughput: Benchmarked at **> 260,000 tokens/second** on multi-core CPU.*

---

### 4. CLI Usage

#### Encode
```bash
indic-tokenizer encode "ನಮಸ್ಕಾರ ಪ್ರಪಂಚ" --tokenizer-dir artifacts/slot_demo --json
```

#### Decode
```bash
indic-tokenizer decode 4509 4527 4538 4763 5 --tokenizer-dir artifacts/slot_demo
```

#### Run Comprehensive Multi-Language Benchmark
```bash
indic-tokenizer benchmark --tokenizer-dir artifacts/slot_demo
```

#### Build a New Tokenizer Artifact
```bash
indic-tokenizer build \
    --out artifacts/v128k \
    --slots-config config/slots.v1.yaml \
    --vocab 128000
```

---

## 7. Automated Test Suite

Run the full verification suite (180 tests covering exact roundtrips, adversarial boundary cases, HuggingFace wrapper, and sharding pipelines):
```bash
pytest -v
```

---

## 8. License

Licensed under the **Apache License, Version 2.0**. See `LICENSE` for details.

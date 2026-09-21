# Diagnostics

Evidence behind the design changes in `SPEC.md`. Each script is standalone and
re-runnable; they read the slot corpora written by `python -m tokenizer.build`.

Run any of them from the repo root after a build exists:

```
python diagnostics/diagnose_slot_training.py
```

---

## 1. `diagnose_slot_training.py` — Unigram loses to BPE on Indic slots

`SPEC.md` originally specified **Unigram** for Devanagari and Kannada, on the
reasoning that agglutinative, morphologically rich scripts suit probabilistic
segmentation better than frequency-greedy BPE. That was an unvalidated assertion.
It is false for this training regime.

Trained on the same 2,032 Devanagari runs, measured as tokens to encode words
that are present in the training data:

| Config | vocab | `नमस्ते` | `भारत` | `कंप्यूटर` | `विद्यार्थी` |
|---|---|---|---|---|---|
| Unigram, alphabet, default pruning | 482 | 4 | 3 | 7 | 7 |
| Unigram, alphabet, pruning disabled | 482 | 4 | 3 | 7 | 7 |
| Unigram, no alphabet | 273 | 4 | 3 | 7 | 7 |
| Unigram, 20× repeated corpus | 561 | 4 | 3 | 7 | 7 |
| Unigram, longer joined sequences | 550 | 2 | 1 | 3 | 4 |
| **BPE, alphabet** | **515** | **1** | **1** | **1** | **1** |

BPE encodes every probe word as a single token. Unigram never does, under any
setting tried. Joining runs into longer sequences helps Unigram but does not
close the gap.

**Conclusion:** BPE for the Indic slots. Implemented in `tokenizer/slots.py`.

**Superseded in magnitude by section 11.** The direction of this finding was
correct and has since been confirmed on real corpora, but *every number above is
an artefact of the corpus it was measured on*: 2,032 runs containing only 75
distinct Devanagari word forms. "3-7 tokens per word" is not what happens on real
text. On 11M characters of Hindi and 60M of Kannada the real gap at our settings is
**+2.4% to +6.7%**. The mechanism identified here (whitespace-free word runs leave
Unigram's EM without context) is the real explanation, and it is why the synthetic
result was so extreme.

## 2. `diagnose_unigram_maxpiece.py` — rules out the obvious explanation

HF's `UnigramTrainer` defaults to `max_piece_length=16` counted in
*pre-tokenized* characters. Under `ByteLevel` one Devanagari letter is 3 such
characters, so a 6-letter word is 18 characters and would be structurally
unrepresentable as one piece — an attractive explanation for the above.

It is wrong. Sweeping `max_piece_length` over 16 / 24 / 32 / 64 changes nothing:
the vocabulary pins at 482 and every probe keeps its token count. `हिंदी` is 15
byte-chars, inside even the default limit, and still fragments into 3.

**Conclusion:** the failure is in convergence, not a length cap.

## 3. `diagnose_whitespace.py` — whitespace ownership costs ~10% on Latin

Under `whitespace_ownership: neutral`, whitespace belongs to the NEUTRAL slot.
Every script slot is therefore trained on whitespace-free word runs and can never
learn a piece spanning a space. Standard BPE learns `" the"`, `" of"`,
`"in the"`; this design forbids it by construction, and additionally every
whitespace run costs at least one full token.

Encoding 236 Latin-dominant documents:

| Scheme | tokens | tokens/char |
|---|---|---|
| A — `neutral` (current default) | 7,383 | 0.2921 |
| B — `leading` (spaces in the run) | 6,618 | 0.2618 |

**10.36% better under `leading`.** The per-sentence view is starker:

```
"The quick brown fox jumps over the lazy dog."
  A: ['The',' ','quick',' ','brown',' ','fox',' ','jumps',' ','over',' ','the',' ','lazy',' ','dog','.']   18 tokens
  B: ['The ','quick brown fox jumps over the lazy dog','.']                                                  3 tokens
```

There are only ~15 distinct English sentences in the synthetic corpus, so scheme
B's absolute numbers are inflated by memorisation. The structural cost of scheme
A — one token per space, and no cross-word merges — is not an artefact and is the
classic reason SentencePiece uses `▁` rather than a separate space token.

**Conclusion:** `leading` is the better default — but the 10.36% figure above is
not reliable. See section 5. It was measured on training documents drawn from ~15
distinct English sentences, so scheme B's advantage was mostly memorisation of
whole sentences. The structural argument holds; the number does not. `leading` was
subsequently implemented end to end and re-measured properly.

---

## 4. `diagnose_byte_tax.py` — every slot pays 256 rows before learning anything

Each trained slot seeded the full 256-character ByteLevel alphabet via
`initial_alphabet`, so it could represent any input losslessly on its own. That
costs **256 vocabulary ids per slot**, before a single learned subword.

| Build | total vocab | rows that are raw bytes | learned rows |
|---|---|---|---|
| leading, target 8000 | 6622 | 1536 (23.2%) | 5081 |
| leading, target 1500 | 1731 | 1536 (**88.7%**) | **190** |
| neutral, target 1500 | 1661 | 1536 (**92.5%**) | **120** |

At a constrained budget nearly the whole vocabulary is byte fallback — roughly
**38 learned subwords per slot**. That is why the partitioned design only beat the
baseline when handed 6622 rows, and collapsed at matched vocabulary.

**Fix, measured.** Each slot's corpus uses only 40–59 distinct bytes:

```
slot              runs   distinct bytes   of 256
NEUTRAL           7881               40    15.6%
LATIN             4532               40    15.6%
DEVANAGARI        1904               43    16.8%
KANNADA           1472               43    16.8%
CODE_MATH          147               59    23.0%
sum of observed bytes across slots :  225
tax under the old rule (256 x 6)   : 1536
avoidable saving                   : 1311 rows
```

Seeding each slot with only its observed bytes (`alphabet_scope=observed`, now the
default) cut the tax from 1536 to 675 rows — 256 of which is the single shared
CATCHALL region — taking learned rows from ~190 to **1533**. Verified by rebuild.

The fix also exposed a **silent data-loss bug**: with `unk_token=None`, HF's BPE
does not merely avoid emitting `<unk>`, it *discards* characters outside the
alphabet without producing any id, so `"\n\n\t"` decoded to `"\n\n"`. Seeding all
256 bytes had masked this completely. Fixed by setting `unk_token="<unk>"`, which
turns an out-of-alphabet character into a detectable special id and routes it to
the lossless CATCHALL fallback. Guarded by `test_bpe_models_have_unk_token_set`.

## 5. Whitespace ownership — the full evaluation history

This took three attempts, because the first two evaluations were invalid. The
failures are the instructive part.

| Evaluation | Corpus | Result |
|---|---|---|
| 1. Training documents | v1 generator, ~20 sentences/lang | `leading` +10.4% |
| 2. Word-shuffled held-out | v1 generator | `neutral` better on every script |
| 3. **Natural held-out** | **v2 generator, 128M-sentence space** | **`leading` better on every script** |

**Evaluation 1 was memorisation.** With ~20 distinct sentences per language, a slot
with a few hundred dedicated tokens could memorise every sentence it saw. Under
`leading` runs are longer, so it memorised whole sentences:
`DEVANAGARI runs=425 tokens=425 bytes/tok=59.6` — one token per sentence.

**Evaluation 2 over-corrected.** Shuffling words removes sequence memorisation but
also destroys exactly the multi-word regularities `leading` exists to capture. It is
structurally biased toward word-level models, i.e. toward `neutral`.

**Evaluation 3 is sound.** The generator was rebuilt (v2) to compose sentences
combinatorially — 128M possible English sentences, 812k Hindi, 423k Kannada — so a
different seed gives genuinely novel yet still-natural text, and word shuffling is
no longer needed.

Natural held-out result, matched corpus, baseline vocabulary 1442, with a measured
**memorisation gap** (train improvement − held-out improvement):

| Script | `neutral` | `leading` | `leading` memo gap |
|---|---|---|---|
| DEVANAGARI | −61.9% | **+24.4%** | 42.5 |
| KANNADA | +8.3% | **+47.6%** | 29.1 |
| LATIN | −65.9% | **+18.6%** | 17.1 |
| CODE_MATH | +91.1% | **+94.1%** | 0.9 |

`leading` is now the default. The large positive memorisation gaps show that even
now, train-side numbers alone would have been badly misleading.

## 6. Runaway structural spans — a bug that produced a confident wrong verdict

The first bits-per-byte run reported the partitioned tokenizer at `0.3242`
tokens/byte against the baseline's `0.1434`, and a decisive loss of `-0.4071 bpb`.
Both numbers were artifacts.

The fence pattern was `r"```[\s\S]*?(?:```|$)"`. The `|$` alternative let a
delimiter with no closing pair match **to the end of the input**. The bpb harness
was encoding one giant concatenated corpus string, so a single unterminated
delimiter routed the remainder of the corpus — megabytes of Devanagari prose —
into `CODE_MATH`. That slot's alphabet had been seeded from code bytes only, so it
raised `UnrepresentableRun` and fell back to per-byte encoding, inflating the
token count ~4.5x. The shared-vocabulary baseline does not segment at all, so it
was untouched, and the comparison silently became "correct tokenizer vs broken
one". Re-measured after the fix: `0.0731` vs `0.1383`.

Two independent defects, both fixed:

1. **Delimiters must close.** `|$` is gone; an unterminated delimiter is simply not
   structural. Truncated code blocks are now treated as prose, which is safe and
   predictable. Guarded by `test_unterminated_fence_does_not_swallow_the_rest` and
   three sibling tests. The old form was also quadratic on stray backticks.
2. **The harness encoded a concatenation, not documents.** Structural detection is
   per document by design. `tokenizer/bpb.py` now encodes document by document, so
   the token numerator and the byte denominator cover exactly the same text.

Lesson: the healthiest-looking result in this project (`decisive`, tiny error bars,
three seeds agreeing to 0.0005) was the wrongest one. Agreement across seeds says
nothing about whether the measurement is valid.

## 7. Whole-run fallback — one bad byte cost an entire sentence

With fallback firing on `UnrepresentableRun` for the **whole run**, a single
character outside a slot's alphabet re-encoded everything containing it through
per-byte CATCHALL. Under `leading`, runs are whole sentences, so:

```
DEVANAGARI run, 2801 chars, encoded standalone : 2400 tokens
same run with ONE leading newline              : 7601 tokens   (3.2x)
```

The newline was absent from the Devanagari slot's observed alphabet, because the
synthetic Devanagari corpus has no newlines in its runs. That is not an exotic
case: `observed`-only alphabets omit whatever the corpus happens to lack, and real
input does not respect training-data boundaries.

Fixed two ways, in order of importance:

1. **Fallback is now per character.** A failing span is bisected until the
   offending characters are isolated, so each costs only its own bytes. Work is
   `O(k log n)` in the number of bad characters `k`. The run above now costs 2420
   tokens with **3** fallback characters (the three backticks). This makes
   robustness a property of the design rather than a property of the alphabet
   heuristic.
2. **`observed+ascii` is the default alphabet** — observed bytes plus whitespace
   and printable ASCII. DEVANAGARI goes from 44 to 141 rows instead of 256, and
   common out-of-corpus characters (newline, digits, quotes) no longer fall back
   at all.

## 8. Bits per byte — the decider

The measurement the objective was waiting for: a small transformer (4 layers,
`d_model=128`, tied embeddings, `seq_len=128`, batch 16, 400 steps = 819,200
training tokens) trained on each tokenizer's output, scored on sequence-novel
held-out text (1,126 documents, 190,351 bytes from a disjoint generator seed).

Five configurations, each replicated in a separate OS process:

| # | budget | partitioned vocab | baseline vocab | baseline algo | params | held-out tokens/byte (part / base) | bpb part | bpb base | **delta** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | natural | 7281 | 7281 | bpe | equal | 0.0731 / 0.0655 | 0.5349 | 0.5606 | **+0.0257** |
| 2 | natural | 7291 | 7285 | bpe | equal | 0.0730 / 0.0655 | 0.5360 | 0.5533 | **+0.0173** |
| 3 | natural | 7291 | 5392 | unigram (pruned) | +16% | 0.0731 / 0.1383 | 0.5262 | 0.5354 | +0.0092 |
| 4 | matched | 1544 | 1544 | bpe | equal | 0.3076 / 0.1423 | 0.5908 | 0.4303 | **−0.1605** |
| 5 | matched | 1544 | 1544 | unigram | equal | 0.3072 / 0.1585 | 0.5809 | 0.4965 | −0.0844 |

Positive delta means the partitioned tokenizer wins.

**Two robust conclusions**, each with the sign repeated in an independent process:

1. **At its natural vocabulary and equal parameter count, the partitioned
   tokenizer wins bits-per-byte** (+0.017 and +0.026, rows 1–2). The margin is
   2–3x the ~0.01 cross-build noise floor from §9.
2. **At matched vocabulary it loses badly** (−0.084 and −0.161, rows 4–5), by
   8–16x the noise floor.

The design is therefore **capacity-hungry**: it needs roughly 4.7x the vocabulary
of a shared model to be ahead, because six slots each carry a byte seed and their
own subwords while a shared vocabulary spends its whole budget on one
distribution.

One detail worth keeping: in rows 1–2 the parameters are *identical*, and the BPE
baseline actually emits **fewer** tokens per byte than the partitioned tokenizer
(0.0655 vs 0.0731) — yet has worse bits-per-byte. So the natural-budget win is
about the *quality* of the units, not about token count. The earlier claim that
partitioning is "1.9x more token-efficient" (row 3) was an artifact of comparing
against a Unigram baseline that had pruned itself down to 5,392 of 7,291 requested
rows; against a baseline that fills its vocabulary, partitioning is slightly
*less* token-efficient and still wins bpb.

Caveats: this is a 1.7M-parameter model on 1.1 MB of synthetic text, so the
*margins* are fragile even though the signs are stable. Real corpora (O1) are
still required. The infrastructure, protocol and correctness tests are in place;
only the data is not.

---

## 9. Vocabulary training is not reproducible across processes

Discovered while checking why the same configuration produced `vocab_size` 7291,
7285 and 7297 on identical input.

| Scope | Result |
|---|---|
| 3 builds in **one** process | identical: `vocab_size=7297`, identical token counts |
| 3 builds in **separate** processes | `vocab_size` 7292 / 7281 / 7283, token counts 23891 / 23881 / 23876 |
| + `PYTHONHASHSEED=0` | still varies: 7294 / 7288 / 7284 |
| + `RAYON_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false` | still varies: 7284 / 7280 / 7281 / 7290 |

The cause is inside the Rust trainer: tie-breaking among equally-ranked merges
depends on hash-map iteration order, which Rust seeds randomly per process.
`PYTHONHASHSEED` does not reach it and neither does thread pinning.

Two consequences, both acted on:

1. **The frozen-artifact design is load-bearing, not decoration.** You cannot
   rebuild a bit-identical vocabulary from a config, so the artifact must be
   shipped and verified by checksum. That is exactly what
   `tokenizer_v1.manifest.json` does, and it is now the only way to reproduce a
   training run.
2. **Small bits-per-byte margins need multiple builds.** Cross-build variation in
   partitioned bpb is about 0.01 -- the same size as the first measured
   "win" (`+0.0092`). `tokenizer/bpb.py` therefore takes `--builds N` and reports
   the delta across builds, and a result is only called decisive if the sign is
   consistent across every build.

## 10. `build()` silently replaced the corpus

`build()` regenerated the corpus whenever the existing one was synthetic, using
whatever `target_words` it was called with. A 4,000-word smoke run therefore
overwrote a 120,000-word corpus in place. Worse, it meant the tokenizer could be
trained on different text than the caller had loaded for evaluation, with nothing
in the output to indicate it.

The corpus file is now authoritative: it is read if present, generated only if
absent, and a mismatch between the stored `target_words` and the requested one is
reported rather than obeyed. `--rebuild-corpus` is the only way to regenerate.
Guarded by `test_build_does_not_clobber_an_existing_corpus`.

## 11. Re-testing Unigram vs BPE on real Indic corpora

Section 1 concluded BPE for the Indic slots. That conclusion rested on a synthetic
corpus holding **75 distinct Devanagari runs (~101 distinct Hindi word forms)** --
close to a worst case for Unigram, whose EM needs lexical diversity. The follow-up
that "repeated the corpus 20x" added occurrences, not words.

Re-run on real text:

| language | source | train | held-out |
|---|---|---|---|
| Hindi | Wikipedia (HF datasets-server) | 11.2M chars, 2,504 articles | 0.93M chars, 279 articles (article-level split) |
| Kannada | local `culturax_kn.jsonl` | 60.5M chars, 19,531 articles | 12.0M chars, 13,664 docs (`E:\kannada_validation\heldout.jsonl`) |

Lexical diversity, at only 2M characters of each:

| corpus | distinct word forms | distinct runs (neutral / leading) | mean run length |
|---|---|---|---|
| synthetic (the original test) | ~101 (Hindi) | 75 | 3.8 chars |
| real Hindi | **31,009** | 31,009 / 45,377 | 4.3 / 35.3 chars |
| real Kannada | **61,146** | 61,146 / 37,926 | 7.0 / 40.4 chars |

300-600x more lexical diversity than the corpus the original decision was made on.

**Held-out tokens per word, lower is better. BPE wins every row.**

| vocab | train chars | language | mode | BPE | Unigram | BPE advantage |
|---|---|---|---|---|---|---|
| 8k | 2M | Hindi | neutral | 1.327 | 1.564 | +17.8% |
| 8k | 2M | Hindi | leading | 1.478 | 1.546 | +4.6% |
| 8k | 2M | Kannada | neutral | 1.814 | 2.207 | +21.7% |
| 8k | 2M | Kannada | leading | 2.135 | 2.277 | +6.7% |
| 8k | 10M | Hindi | neutral | 1.277 | 1.486 | +16.4% |
| 8k | 10M | Hindi | leading | 1.429 | 1.463 | **+2.4%** |
| 8k | 10M | Kannada | neutral | 1.745 | 2.129 | +22.0% |
| 8k | 10M | Kannada | leading | 2.070 | 2.180 | +5.3% |
| 8k | 30M | Kannada | neutral | 1.720 | 2.112 | +22.8% |
| 8k | 30M | Kannada | leading | 2.046 | 2.141 | +4.6% |
| 32k | 5M | Hindi | neutral | 1.116 | 1.450 | +30.0% |
| 32k | 5M | Hindi | leading | 1.133 | 1.298 | +14.6% |
| 32k | 5M | Kannada | neutral | 1.390 | 1.962 | +41.1% |
| 32k | 5M | Kannada | leading | 1.647 | 1.921 | +16.6% |

The `neutral` and `leading` columns are **not** comparable to each other: the
measurement counts only the script slot's tokens, so under `neutral` the space
tokens (owned by the NEUTRAL slot) are excluded while under `leading` they sit
inside the run. BPE vs Unigram *within* a mode is fair, since both see identical
runs.

### What this changes

1. **The decision stands.** BPE wins all 14 configurations across two languages,
   2M-30M characters of real text, 8k and 32k vocabularies, both whitespace modes.
2. **The magnitude was badly overstated.** "3-7 tokens where BPE uses 1" was an
   artefact of a corpus with ~101 distinct words. At the settings we actually use
   (8k, `leading`) the real gap is **+2.4% to +6.7%**.
3. **More data helps Unigram only under `leading`, and slowly**: +6.7% -> +5.3% ->
   +4.6% at 2M / 10M / 30M Kannada characters. Under `neutral` the gap is flat near
   22% and more data does not help at all, which confirms the mechanism: short
   whitespace-free runs give Unigram's EM almost no context.
4. **A larger vocabulary hurts Unigram rather than helping it.** At 32k the gap
   widens to +14.6% / +16.6% (`leading`) and +30% / +41% (`neutral`), and Unigram
   failed to fill the requested budget in two of the 32k runs (Hindi `neutral`
   reached only 21,842 of 32,000). The intuition that more pieces favours a
   probabilistic segmenter does not hold here.
5. **`leading` is vindicated again.** It improves both algorithms substantially and
   narrows the algorithm gap, consistent with whitespace ownership being the
   dominant design variable.

Open question this could not settle: whether the `leading` gap keeps closing past
30M characters. Kannada has 60M available locally, Hindi only 11M. On the observed
trend it would close slowly, if at all.

Reproduce with:

```
python diagnostics/fetch_indic_corpus.py --langs hi --target-chars 20000000
python diagnostics/extract_jsonl_text.py --source "G:/jnana_raw/culturax_kn.jsonl" \
    --out data_real/kn_train.jsonl --target-chars 60000000
python diagnostics/compare_indic_algorithms.py --train-chars 2000000 \
    --heldout-chars 1000000 --vocabs 8000 --langs hi,kn
```

## 12. Vachana's Kannada tokenizer: what actually beat what

A sibling project (`E:\VachanaLLM`) trained both `kannada_32k_unigram` and
`kannada_32k_bpe` with SentencePiece and picked Unigram. Their report
(`tokenizer/TOKENIZER_REPORT.md`, `eval_results.json`) reads:

| model | vocab | fertility (Kannada-only) | vocab utilisation |
|---|---|---|---|
| kannada_32k_unigram | 32,000 | **1.7295** | 79.7% |
| kannada_32k_bpe | 32,000 | **1.7536** | 81.6% |
| sarvam-1 (shared, **10 Indic languages**) | 68,096 | **2.4906** | **24.2%** |

Two very different numbers are in play and they are easy to conflate:

- Unigram vs BPE: **(1.7536 − 1.7295) / 1.7536 = 1.37%**. Not a comfortable win.
- Dedicated vs sarvam-1: **(2.4906 − 1.7295) / 2.4906 = 30.6%**. That is the
  comfortable win, and it is against a tokenizer whose vocabulary is **shared
  across ten Indic languages**. sarvam-1 also burns 1.40% of Kannada tokens on
  byte fallback versus 0.07%, and only 24.2% of its 68k vocabulary ever fires on
  Kannada — three quarters of its embedding budget is dead weight for this use.

So the large effect is **dedicated vocabulary vs shared vocabulary**, exactly as
the Vachana plan predicted. It is not an algorithm effect, and it is not a
two-languages-versus-many-languages effect at the algorithm level.

### Cross-implementation head-to-head

All on identical Kannada held-out runs (44,332 runs, 1,916,388 chars, 240,049
words), 32k vocabulary, SentencePiece settings copied from Vachana's report:

| tokenizer | implementation | tokens/word |
|---|---|---|
| sp_bpe_mine | SentencePiece | **1.6181** |
| vachana_sp_unigram | SentencePiece | 1.6257 |
| hf_bpe_mine | HF tokenizers | 1.6470 |
| vachana_sp_bpe | SentencePiece | 1.6526 |
| sp_unigram_mine | SentencePiece | 1.6577 |
| hf_unigram_mine | HF tokenizers | **1.9210** |
| sarvam1_shared_10lang | SentencePiece | 2.5082 |

Three findings:

1. **HF's `UnigramTrainer` is materially weaker than SentencePiece's.**
   `hf_unigram_mine` 1.9210 vs `sp_unigram_mine` 1.6577 on identical training runs
   and identical held-out — a **15.9%** gap. HF BPE and SentencePiece BPE agree
   closely by contrast (1.6470 vs 1.6181, 1.8%).
2. **For Kannada, Unigram vs BPE is 1-3% and the sign is not stable.**
   Vachana measured Unigram better by 1.4%; I measure BPE better by 2.4% with
   SentencePiece and by 16.6% with HF. Under SentencePiece — the fair comparison —
   the two algorithms are effectively tied.
3. **The algorithm choice is dominated by the implementation choice.** 15.9%
   separates two Unigram trainers; 1-3% separates the two algorithms.

### What this corrects in my own work

My round-2/3 conclusion "BPE beats Unigram for Indic, and a larger vocabulary
makes it worse" was measured entirely with HF `tokenizers`. The direction for
Kannada is not reproducible under SentencePiece, and the 32k result in
particular was inflated by HF's weak Unigram. The claim should be stated as:

- With HF `tokenizers`: use BPE. HF's Unigram is meaningfully worse, and no
  amount of data fixed it.
- With SentencePiece: for Kannada the two are a wash. Unigram has a mild
  linguistic argument (Vachana's probe table shows clean root/suffix boundaries),
  and BPE measured marginally better on my held-out. Neither difference is large
  enough to justify a strong claim.
- Either way, the design decision that matters is **dedicated per-script
  vocabulary versus a shared multilingual one**, worth ~30%, which is what this
  project's slot model already does.

Independent corroboration, worth noting: Vachana hit the same whitespace bug
documented in §7. SentencePiece's `remove_extra_whitespaces` defaults to `True`
and silently collapsed whitespace runs, giving them 97.6% round-trip losslessness
until they set `remove_extra_whitespaces=False`. Two projects, two different
stacks, the same silent-reconstruction-failure class.

Reproduce with:

```
python diagnostics/sp_vs_hf_indic.py --train-chars 5000000 --heldout-chars 2000000 --vocab 32000
```

## 13. Provider benchmark: the algorithm was never the decision

Sections 11 and 12 concluded BPE for Indic. **That was wrong, and so was the
correction that produced it.** The real variable is a single trainer setting.

### Provider landscape

Checked against PyPI at run time. Only two maintained options can *train* a
subword vocabulary:

| package | last release | trains BPE | trains Unigram |
|---|---|---|---|
| `tokenizers` (HF, Rust) | 2026-09-03 | yes | yes |
| `sentencepiece` (Google, C++) | 2026-07-12 | yes | yes |
| `tiktoken` | 2026-08-17 | **inference only, no trainer** | no |
| `turbobpe` | 2026-08-11 | minbpe-style, no byte-level | no |
| `youtokentome` | 2020 | abandoned | no |
| `subword-nmt` | 2021 | abandoned | no |
| `fastbpe` | 2019 | abandoned | no |

SuperBPE (COLM 2025) and `minbpe` are not packaged; `rustbpe` fills tiktoken's
missing trainer but targets GPT-style regex pre-tokenization, which is
English-centric. So the benchmark is HF `tokenizers` vs SentencePiece.

### The bug: `max_piece_length` is a byte budget, not a character budget

SentencePiece's `max_sentencepiece_length` defaults to 16 *Unicode characters*, so
it can learn a 16-akshara Kannada word. HF's `UnigramTrainer.max_piece_length`
defaults to 16 *pre-tokenized tokens*, and under a ByteLevel pre-tokenizer one
Indic akshara is ~3 of them -- so HF was capped at roughly **5 aksharas**. The
median Kannada training run is 88 byte-chars, so the default truncates almost
everything.

Sweeping it on real Kannada, 32k vocab:

| `max_piece_length` | 16 (default) | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|
| tokens/word | 1.9210 | 1.5177 | **1.4965** | 1.4979 | 1.4989 |

48 is the knee. One parameter is worth **22%**.

### Benchmark: 5 configurations x 2 languages x 2 vocabularies

Identical runs, identical held-out, all verified lossless:

| language | vocab | winner | tok/word | runner-up | margin |
|---|---|---|---|---|---|
| Hindi | 8k | **hf_unigram_mpl48** | 1.2977 | hf_bpe 1.4472 | 10.3% |
| Hindi | 32k | **hf_unigram_mpl48** | 1.0589 | hf_bpe 1.1333 | 6.6% |
| Kannada | 8k | **hf_unigram_mpl48** | 1.8613 | sp_bpe 2.0149 | 7.6% |
| Kannada | 32k | **hf_unigram_mpl48** | 1.4965 | sp_bpe 1.6181 | 7.5% |

Full ordering, Kannada at 32k: `hf_unigram_mpl48` 1.4965 < `sp_bpe` 1.6181 <
`hf_bpe` 1.6471 < `sp_unigram` 1.6577 < `hf_unigram_default` 1.9210.

**HF Unigram with a raised piece limit wins every configuration.** HF BPE and
SentencePiece BPE sit within 2% of each other at every budget -- the BPE
implementations agree, and it is the Unigram *configuration* that varied. HF
Unigram with `max_piece_length=48` is now configured in `tokenizer/slots.py` as
the Indic algorithm.

### The arc, recorded honestly

The specification originally said Unigram for Indic. I "corrected" it to BPE on a
synthetic corpus with 75 distinct Devanagari runs, then re-affirmed BPE on real
data -- where HF's Unigram was still running at its default. Both times I measured
a configuration artefact and reported an algorithmic result.

- First error: corpus too small for Unigram's EM -> "Unigram is 3-7x worse".
- Second error: HF's byte-vs-character piece limit -> "HF Unigram is 15.9% worse".

The original instinct was right. The lesson is not "trust first instincts" but
that **an algorithm comparison is not valid until both sides are configured
properly and evaluated on real text.**

### Corroboration from the literature

SuperBPE (COLM 2025, [paper](https://arxiv.org/abs/2503.13423),
[code](https://github.com/PythonNut/superbpe)) reports that removing BPE's
whitespace pre-tokenization restriction lets tokens span words, cutting tokens by
up to 33% at fixed vocabulary, with +4.0% average downstream improvement. That is
independently the same variable this project found dominant in section 5:
**whitespace ownership**, not the merge algorithm. Our `leading` mode already
permits cross-word pieces within a script run, which is the SuperBPE effect in a
multilingual-slot setting. Worth revisiting once real corpora are wired in.

Reproduce with:

```
python diagnostics/benchmark_indic_tokenizers.py --train-chars 5000000 \
    --heldout-chars 2000000 --vocabs 8000,32000 --langs kn,hi
python diagnostics/tune_hf_unigram.py
```

## 14. Multi-language benchmark: 12 Indic languages plus romanized code-mixes

Section 13 picked Unigram with `max_piece_length=48` from two languages. Two
languages cannot support a design decision, so this extends it to every major
Indic script and to romanized code-mixed Latin.

### Corpora

The local Indic files turned out to be **Kannada-only** -- both
`indiccorp_v2.jsonl` (21.5 GB) and `sangraha_verified.jsonl` (20.0 GB) begin at
`group=kannada` and stay there for the first 4M lines, because the Jnana/Vachana
project is Kannada-focused. So the other scripts were fetched from Wikipedia via
the HF datasets-server (~2M chars each):

| script | languages |
|---|---|
| DEVANAGARI | Hindi, Marathi, Nepali |
| BENGALI | Bengali |
| TAMIL | Tamil |
| TELUGU | Telugu |
| MALAYALAM | Malayalam |
| GUJARATI | Gujarati |
| GURMUKHI | Punjabi |
| ORIYA | Odia |
| ARABIC | Urdu |
| KANNADA | Kannada (60M chars, local CulturaX) |

Romanized code-mixed: **Hinglish** (27,797 utterances, 1.5M chars) and
**Tanglish** (12,300 utterances, 1.5M chars) from the Hub. **Kanglish has no
public corpus** -- searches for `kanglish`, `kannada romanized`,
`kannada english mixed` and `kannada transliteration` all returned nothing.
Tanglish is the closest available proxy (romanized Dravidian + English) and is
used in its place; that substitution is a real gap, not a solved problem.

### A bug worth noting

The first run reported **0 runs for Bengali, Tamil, Telugu, Malayalam, Gujarati,
Gurmukhi, Odia and Urdu**. Cause: the extraction filtered runs by slot name, but
`DEFAULT_SLOT_MAP` only routes COMMON/LATIN/DEVANAGARI/KANNADA -- every other
script falls through to CATCHALL. So eight of twelve languages silently produced
an empty corpus and the run looked like it had covered them. Building a slot map
with one slot per script fixed it. Worth recording because the failure was
*silent*: had the skip guard not existed, the benchmark would have reported
"12 languages" while measuring four.

### Result A: one vocabulary per language

Held-out tokens/word, 16k vocab, ~1.5M chars training each:

| language | script | BPE | Unigram | winner | margin |
|---|---|---|---|---|---|
| Bengali | BENGALI | 1.5066 | **1.4310** | unigram | 5.0% |
| Tamil | TAMIL | 1.8945 | **1.7818** | unigram | 5.9% |
| Kannada | KANNADA | 1.9498 | **1.9228** | unigram | 1.4% |
| Gujarati | GUJARATI | 1.5150 | **1.4791** | unigram | 2.4% |
| Oriya | ORIYA | 1.4537 | **1.4398** | unigram | 1.0% |
| Malayalam | MALAYALAM | 2.3243 | **2.3174** | unigram | 0.3% |
| Hindi | DEVANAGARI | **1.3211** | 1.3268 | bpe | 0.4% |
| Marathi | DEVANAGARI | **1.7006** | 1.7015 | bpe | 0.1% |
| Nepali | DEVANAGARI | **1.5242** | 1.5438 | bpe | 1.3% |
| Telugu | TELUGU | **1.7410** | 1.7600 | bpe | 1.1% |
| Punjabi | GURMUKHI | **1.1142** | 1.1490 | bpe | 3.0% |
| Urdu | ARABIC | **1.0583** | 1.1393 | bpe | 7.1% |

Unigram wins 6, BPE wins 6. **Every margin is under 7% and the sign is
language-dependent.** Per-language at this budget the two algorithms are
practically interchangeable.

### Result B: one vocabulary per SCRIPT, shared across its languages

This is the design's actual configuration. Pooling Hindi + Marathi + Nepali into
a single Devanagari slot:

| language | BPE | Unigram | winner | margin |
|---|---|---|---|---|
| Hindi | 1.3811 | **1.3053** | unigram | 5.5% |
| Marathi | 1.8455 | **1.7009** | unigram | 7.8% |
| Nepali | 1.7003 | **1.5796** | unigram | 7.1% |

**When the vocabulary is shared across languages, Unigram wins all three, by
5.5-7.8%.** Note also the direction of the sharing cost: for BPE every language
gets worse when pooled (Hindi 1.3211 -> 1.3811, Marathi 1.7006 -> 1.8455), while
for Unigram Hindi actually *improves* (1.3268 -> 1.3053) and Marathi is flat,
because Unigram benefits from the extra 4.4M characters of training data while
BPE's greedy merges get diluted across languages.

This is the answer to "does it matter that we have many languages": **yes, and it
favours Unigram.** The design shares one slot per script, so the shared case is
the one that governs.

### Result C: the LATIN slot with code-mixed text

English + Hinglish + Tanglish pooled into one LATIN slot (193,361 runs, 5.2M
chars), which is what the encoder must do since it cannot separate romanized
Indic from English:

| held-out source | BPE | Unigram | winner | margin |
|---|---|---|---|---|
| English | 1.4294 | **1.3155** | unigram | 8.0% |
| Hinglish | 1.0322 | **0.9320** | unigram | 9.7% |
| Tanglish | 1.7404 | **1.6476** | unigram | 5.3% |

Unigram wins all three, so **LATIN moves to Unigram as well.** The margin is
largest on Hinglish, the code-mixed case, which is the opposite of the worry that
a shared Latin slot would hurt romanized text.

### Vocabulary-size caveat

Hindi per-language measured Unigram ahead by 6.6% at 32k vocab / 5M chars (§13)
but behind by 0.4% at 16k / 1.5M chars here. Vocab size and training-data size
both changed between those runs, so they cannot be separated from this data. The
Unigram advantage appears to grow with both. The shared-slot results above used
the same 16k budget for both algorithms, so they are internally comparable; the
cross-budget comparison is not.

### Cost

Unigram training is **~10x slower** than BPE at these sizes: 50-135s versus 3-9s
per slot for the same corpus and vocabulary. For Indic scripts whose corpora are
in the tens of MB this is acceptable; for a full 100M-character corpus it becomes
a real budget item, and it is the main argument for BPE if iteration speed ever
dominates.

### Final configuration

`NEUTRAL` and `CODE_MATH` stay BPE -- no benchmark covers them and BPE is the
conventional choice for punctuation, digits and code. Everything else is Unigram
with `max_piece_length=48`.

Reproduce with:

```
python diagnostics/scan_indic_corpora.py --cap-lines 4000000
python diagnostics/fetch_indic_multilang.py --target-chars 1800000
python diagnostics/fetch_romanized_mix.py --target-chars 1500000
python diagnostics/benchmark_indic_all.py --vocab 16000
python diagnostics/benchmark_latin_mix.py --vocab 16000
```

## 15. Why Unigram wins: it is robustness to vocabulary sharing, not language skill

Section 14 showed Unigram winning on English inside the pooled LATIN slot (+8.0%).
That invited an obvious objection: English is not an Indic language, so why would
Unigram be better at it? Is the win just the pooled vocabulary?

Decomposed, English at both budgets:

| vocab | English ALONE BPE | English ALONE Unigram | Unigram margin alone |
|---|---|---|---|
| 16k | 1.3226 | 1.2957 | **+2.0%** |
| 32k | 1.1907 | 1.1902 | **+0.0%** (tie) |

So **Unigram is not better at English in isolation.** At 32k the two are
identical to four decimal places. The pooled margin must come from pooling, and
it does:

| vocab | tokenizer | English-only | Pooled | cost of sharing |
|---|---|---|---|---|
| 16k | BPE | 1.3226 | 1.4474 | **−9.4%** |
| 16k | Unigram | 1.2957 | 1.3295 | −2.6% |
| 32k | BPE | 1.1907 | 1.2807 | **−7.6%** |
| 32k | Unigram | 1.1902 | 1.2083 | −1.5% |

Also note that `max_piece_length` is irrelevant for English (mpl16 1.2947 vs
mpl48 1.2957 at 16k). That is exactly as predicted: English is one byte per
character, so a limit of 16 already permits 16 characters. The setting only ever
mattered for 3-byte scripts.

### The mechanism generalises

The same comparison on Devanagari, where Hindi + Marathi + Nepali are pooled into
one slot:

| language | BPE alone | BPE shared | **BPE cost** | Unigram alone | Unigram shared | **Unigram cost** |
|---|---|---|---|---|---|---|
| Hindi | 1.3211 | 1.3811 | **+4.5%** | 1.3268 | 1.3053 | **−1.6%** |
| Marathi | 1.7006 | 1.8455 | **+8.5%** | 1.7015 | 1.7009 | −0.0% |
| Nepali | 1.5242 | 1.7003 | **+11.6%** | 1.5438 | 1.5796 | +2.3% |

**BPE loses 4.5-11.6% when a vocabulary is shared; Unigram loses 0-2.6%, and for
Hindi it actually gains.** Across Latin and Devanagari, BPE is roughly 3-5x more
fragile to vocabulary sharing than Unigram.

### What this reframes

The finding is not "Unigram is better for Indic". It is:

> **Unigram's advantage is robustness to a shared vocabulary. For a
> single-language vocabulary the two algorithms are close to interchangeable;
> as soon as a slot holds more than one language, BPE degrades several times
> faster.**

That single statement accounts for every measurement in §13 and §14:

| case | vocabulary | observed |
|---|---|---|
| 12 languages, one vocab each, 16k | single-language | 6 wins / 6 losses, all ≤7% |
| English alone, 32k | single-language | tie (0.0%) |
| Hindi, Kannada alone, 32k | single-language | Unigram +6.6% / +7.5% |
| Devanagari shared (hi+mr+ne) | **shared** | Unigram +5.5% / +7.8% / +7.1% |
| LATIN shared (en+hinglish+tanglish) | **shared** | Unigram +8.0% / +9.7% / +5.3% |

The two single-language exceptions (Hindi and Kannada at 32k) are the cases with
rich agglutinative morphology, where Unigram's global likelihood objective helps
even without sharing — consistent with the original spec rationale.

Why BPE is the fragile one: its merges are allocated greedily by pair frequency,
so in a mixture the dominant language consumes the budget and the remaining
merges go to rare cross-lingual artefacts. Unigram's EM allocates probability
mass across the whole mixture, so it degrades gracefully. This is a plausible
mechanism consistent with the data; it has not been verified directly.

### Consequence for this design

Every slot except `NEUTRAL` and `CODE_MATH` is either shared across languages by
construction (DEVANAGARI, KANNADA is single-language, LATIN is unavoidably shared
because romanized Indic cannot be separated from English) or morphologically
rich. So Unigram is the right default here — but the *reason* matters, and it
means the choice would flip for a genuinely monolingual, analytic-language slot.

Reproduce with:

```
python diagnostics/diagnose_english_unigram.py --vocabs 16000,32000
```

## 16. Choosing vocabulary sizes

Fertility falls **monotonically** with vocabulary size, so "pick the size that
minimises fertility" has no interior optimum — it only ever says "bigger". The
sweep's knee column degenerated to the largest size tested for exactly that
reason, which is worth recording so nobody re-runs it expecting a knee.

Measured on ~1.3M characters of training text, Unigram, `max_piece_length=48`:

| condition | 4k | 8k | 16k | 32k | 64k | 100k |
|---|---|---|---|---|---|---|
| DEVANAGARI (hi+mr+ne) | – | 1.6324 | 1.4748 | 1.3385 | 1.2237 | 1.2163 |
| LATIN (en+hinglish+tanglish) | – | 1.7923 | 1.6284 | 1.4727 | 1.2876 | 1.2876 |
| KANNADA | – | 2.1581 | 1.9452 | 1.7511 | 1.6005 | – |
| BENGALI | 1.6817 | 1.4025 | 1.1813 | 1.0089 | – | – |
| TAMIL | 2.1742 | 1.8444 | 1.5744 | 1.3412 | 1.2425 | – |
| TELUGU | 2.5119 | 2.1569 | 1.8116 | 1.5119 | – | – |
| MALAYALAM | 2.8508 | 2.3682 | 1.9513 | 1.6348 | 1.4696 | – |
| GUJARATI | 1.6748 | 1.4258 | 1.2102 | 1.0273 | – | – |
| GURMUKHI | 1.3908 | 1.1912 | 1.0154 | 0.8572 | – | – |
| ORIYA | 1.5261 | 1.3185 | 1.1438 | 0.9980 | – | – |
| ARABIC (Urdu) | 1.2458 | 1.0570 | 0.8896 | 0.7495 | 0.6528 | – |

Marginal gain per doubling is still **7-16%** at 64k, so nothing here is saturated.
Two measured ceilings matter more than the curve shape:

* DEVANAGARI reached only **68,062** pieces when asked for 100,000.
* LATIN capped at **62,036** and stopped improving entirely (+0.0% from 64k to 100k).

A 1.3M-character corpus simply cannot support those vocabularies. Real corpora
will, so the recommended sizes are deliberately conservative.

### Two criteria that do have structure

**Marginal gain per doubling** — the value of one more doubling:

| condition | 16k→32k | 32k→64k |
|---|---|---|
| Malayalam | +17.6% | +16.2% |
| Urdu | +15.8% | +15.8% |
| Tamil | +14.6% | +14.8% |
| Kannada | +10.0% | +8.6% |
| LATIN | +9.6% | +12.6% |
| DEVANAGARI | +9.2% | +8.6% |
| (then 64k→100k) | | DEVANAGARI +0.6%, LATIN +0.0% |

**Vocabulary utilization** — fraction of pieces that fire on held-out text:
0.75 at 8k, 0.52-0.65 at 16k, 0.28-0.55 at 32k, down to 0.20-0.37 at 64k.

These utilization figures are **underestimates**: they were measured on ~350k
characters of held-out text (30-70k words). A piece that fires once in 100M tokens
is perfectly healthy and will not appear in a 60k-word sample. Vachana's report
sizes its equivalent test at 100M+ tokens; on that scale the same vocabularies
would score far higher.

### What actually decided the recommended sizes

Since fertility cannot pick a size, sizing came from **relative difficulty
measured at a common 32k budget** — directly comparable across scripts — combined
with how many languages each slot carries and the embedding budget. The resulting
table is in `SPEC.md` §6.1.1. The short version: shared slots (`LATIN`,
`DEVANAGARI`) take the ceiling, the still-hard scripts (Malayalam, Kannada,
Telugu, all ≥1.5 tokens/word at 32k) get 48k, and the already-easy ones (Urdu
0.75, Punjabi 0.86, Odia 1.00, Gujarati 1.03) get 20-24k.

If the ~449k total is too large, halving a slot costs roughly its marginal gain
in reverse — 9-16% fertility for that script — and the cheapest cuts are the
low-difficulty slots.

Reproduce with:

```
python diagnostics/sweep_vocab_sizes.py
python diagnostics/choose_vocab_sizes.py
```

## 17. One shared Unigram slot vs per-script slots

The proposal: there are only two algorithms, so have only two slots -- put every
Unigram script into one slot with a shared 256k vocabulary, and leave BPE as-is.

This is **not** the same as §14's shared-Devanagari result. There, three languages
shared one script's byte space. Here Tamil, Bengali, Gurmukhi, Odia and Devanagari
all compete for the same rows while occupying nearly disjoint UTF-8 ranges, so most
rows can only ever fire for one script.

Tested at equal total vocabulary, 10 scripts, ~460k characters of training text
each, Unigram `max_piece_length=48`, per-script budgets following the SPEC v2
proportions. Mean held-out tokens/word, lower better:

| total vocab | A per-script | B one shared | C Indic-shared + separate LATIN |
|---|---|---|---|
| 128,000 | **1.5217** | 1.6592 (**+9.0%**) | 1.6450 (+8.1%) |
| 256,000 | **1.3105** | 1.6221 (**+23.8%**) | 1.5782 (+20.4%) |

Per-script detail at 256,000:

| script | A per-script | B one shared | B vs A |
|---|---|---|---|
| Odia | 1.1451 | 1.6562 | **+44.6%** |
| Telugu | 1.2727 | 1.7825 | **+40.1%** |
| Malayalam | 1.4716 | 1.9831 | **+34.8%** |
| Bengali | 0.9888 | 1.3114 | **+32.6%** |
| LATIN | 1.3800 | 1.8098 | **+31.1%** |
| Tamil | 1.3089 | 1.6661 | +27.3% |
| Gujarati | 1.1619 | 1.3906 | +19.7% |
| Gurmukhi | 0.9618 | 1.0862 | +12.9% |
| Devanagari | 1.4237 | 1.4853 | +4.3% |
| Kannada | 1.9902 | 2.0501 | +3.0% |

### Three things this establishes

1. **The penalty grows with vocabulary size.** 9.0% worse at 128k, 23.8% at 256k.
   At 128k four scripts actually *gained* from sharing (Gurmukhi −7.5%, Devanagari
   −4.1%, Kannada −3.7%, Gujarati −1.3%) because the shared pool gave them more
   rows than their proportional per-script budget. At 256k the per-script budgets
   grew too, that advantage vanished, and **all ten scripts lose**.

2. **Sharing shares competition, not just capacity.** What decides which script's
   pieces survive training is pooled corpus frequency. Scripts whose full-word
   forms are rarer in the mixture (Odia, Telugu, Malayalam) lose most. LATIN loses
   badly too (+31%) because it goes from a dedicated slot to competing with nine
   Indic scripts.

3. **A big shared vocabulary cannot even be filled.** Asked for 256,000 pieces on
   a 4.6M-character pool, the shared model produced **206,760**. Splitting the same
   total across per-script slots reached their targets, because each slot sees a
   concentrated distribution rather than a diluted one.

This matches the closest real-world instance: sarvam-1's 68k vocabulary shared
across 10 Indic languages scored 30.6% worse on Kannada than a dedicated 32k slot,
with 3x worse utilization (§12).

### Where grouping still makes sense

Collapsing `NEUTRAL` and `CODE_MATH` into a single BPE slot is reasonable — both
are ASCII-heavy, small, and adjacent in character distribution — taking the model
from 13 slots to 12 without touching any Unigram vocabulary. What is rejected is
grouping **different scripts** into one Unigram vocabulary.

Caveat: measured on ~4.6M pooled characters. A much larger corpus would fill a
shared 256k vocabulary more completely, and the 256k gap might narrow. The
direction was consistent at both budgets tested and strengthened with size, but a
100M-character rerun is the honest way to close this.

Reproduce with:

```
python diagnostics/test_shared_unigram_slot.py --total 128000
python diagnostics/test_shared_unigram_slot.py --total 256000
```

## 18. Cost of the v3 vocabulary trim

Proposed: keep every language's vocabulary separate (section 17 settled that), but
shrink the Dravidian slots to 32k and trim the minor scripts. Measured at the old
and new size on identical runs, Unigram `max_piece_length=48`:

| slot | old → new | old tokens/word | new tokens/word | **cost** | utilization |
|---|---|---|---|---|---|
| Telugu | 48,000 → 32,000 | 1.3796 | 1.5119 | **+9.6%** | 0.464 → 0.547 |
| Malayalam | 48,000 → 32,000 | 1.4988 | 1.6348 | **+9.1%** | 0.446 → 0.548 |
| Odia | 24,000 → 16,000 | 1.0540 | 1.1438 | **+8.5%** | 0.488 → 0.571 |
| Kannada | 48,000 → 32,000 | 1.6552 | 1.7511 | **+5.8%** | 0.316 → 0.391 |
| Punjabi | 20,000 → 16,000 | 0.9623 | 1.0154 | **+5.5%** | 0.617 → 0.655 |
| Tamil | 40,000 → 32,000 | 1.2905 | 1.3412 | **+3.9%** | 0.440 → 0.494 |

Unchanged slots, measured for reference on the same runs:

| slot | vocab | tokens/word | utilization |
|---|---|---|---|
| LATIN (en+hinglish+tanglish) | 64,000 | 1.2875 | 0.212 |
| DEVANAGARI (hi+mr) | 48,000 | 1.2291 | 0.249 |
| BENGALI (bn+as) | 32,000 | 1.0089 | 0.509 |
| GUJARATI | 24,000 | 1.1002 | 0.565 |

Total vocabulary goes **413,280 → 345,280 (−16.5%)**, or −23.1% against the v1
allocation of 449,280. Embedding parameters at `d_model=2048` tied fall from
~920M to ~707M.

### Two observations

1. **The trim costs most where it hurts most.** Telugu (+9.6%) and Malayalam
   (+9.1%) were already the highest-fertility scripts, so the reduction takes the
   largest toll on the slots least able to absorb it. If any slot is worth
   restoring, it is these two — +32,000 rows between them would recover ~9%.
2. **Utilization rises at every reduced slot**, which is the expected signature of
   a vocabulary that was slightly oversized. Kannada 0.316 → 0.391 and Telugu
   0.464 → 0.547 are both healthier numbers. So the trim is not simply a loss: it
   trades ~4-10% fertility for 16.5% fewer embedding rows and better-utilised
   vocabularies.

Caveat: these are held-out fertility costs, not bits-per-byte. A 9.6% fertility
increase on Telugu does not translate one-to-one into a 9.6% worse model — the
bits-per-byte harness (§8) is the tool for that, and it has not been run on v3.

Reproduce with:

```
python diagnostics/measure_v3_sizes.py
```

## Open

- **The design is capacity-hungry.** At its natural vocabulary (5938) it beats the
  shared baseline on every script. Constrained to the baseline's 1442 rows it loses
  on Devanagari (−44%) and Latin (−100%). Six slots each need a byte seed plus their
  own subwords, while a shared vocabulary concentrates its whole budget on one
  distribution. Whether that trade is worth it depends on whether embedding rows are
  cheap relative to sequence-length compute in the intended model: at `d_model=2048`
  the difference is ~3M vs ~12M parameters, negligible against a ~25% reduction in
  sequence length.
- **bits-per-byte is still unmeasured.** It needs a trained language model and is the
  stated decider; everything here is a tokenizer-level proxy.
- Re-run everything on real corpora. Sentence overlap between the synthetic train and
  held-out splits is still 0.197, driven by the small Hinglish/Kanglish sentence
  spaces (58k / 41k); Devanagari and Kannada spaces are large.

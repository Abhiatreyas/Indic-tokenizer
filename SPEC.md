# Multilingual Script-Partitioned Tokenizer — Design Spec v1

Status: **IMPLEMENTED AND MEASURED — bits-per-byte favours the partitioned design at its natural vocabulary and the shared baseline at matched vocabulary; see §12.5**
Awaiting: real corpus paths (O1), and a decision on O2/O6/O8/O9
Purpose: pretraining a multilingual LLM from scratch
Stack: Python + HuggingFace `tokenizers` / SentencePiece

---

## 1. What this is

A tokenizer whose **id space is partitioned into slots**. A slot is an **algorithm
applied to a set of scripts** — declared as `(name, algorithm, scripts[])`. Each slot is
trained independently. The final vocabulary is the concatenation of the slots, and the
*slot boundaries are the metadata* that tells the decoder which algorithm to use for any
given id.

Because a slot owns a *set* of scripts, several languages share one slot whenever they
share a script and an algorithm. There are no per-language slots anywhere in this design.

Nothing is emitted into the token stream to mark language. The stream is simply a
sequence of ids in original text order; every id is self-describing because it falls
in exactly one slot's id range.

### Non-goals for v1

- Per-language slots within a single script (Hindi vs Marathi vs Nepali are **one**
  slot). This would require language identification, which is deferred — see §3.4.
- Language tags or sentinel tokens in the stream by default. `<|slot_sep|>` exists as an
  opt-in, costing one row — see §3.3.
- `offset_mapping` / token→character spans. Not needed for pretraining; noted as a
  known gap if span-labelled tasks are ever added.
- Model architecture. This spec stops at the tokenizer.

---

## 2. Terminology — read this first

The earlier review used "script", "language", and "slot" loosely and that caused
confusion. They are three different things:

| Term | Meaning | Example |
|---|---|---|
| **Language** | The linguistic language | Hindi, Marathi, English, Hinglish |
| **Script** | The writing system (a Unicode property, mechanically detectable) | Devanagari, Latin, Kannada |
| **Slot** | A contiguous id range in the final vocab: one algorithm applied to a set of scripts | `DEVANAGARI` = (BPE, {DEVANAGARI}) |

**The rule that decides everything: the encoder must be able to choose a slot from the
raw bytes, with no guessing, no model, no probability.**

Script satisfies this — `unicodedata` tells you the script of every character
deterministically. Language does not: Hindi and Marathi are both Devanagari, and
Hinglish is written in Latin script and is character-for-character indistinguishable
from English. Any router that tries to pick between them needs a language-ID model,
which is a source of silent, unrecoverable errors.

So: **slot ≡ (algorithm, set of scripts).** A slot is created when a set of text
shares (a) scripts that can be detected deterministically, and (b) one tokenizer
algorithm. One script may be split across two slots later if measurement shows it
should be (e.g. prose vs code), but never across two *languages* in v1.

This is compatible with your instruction "each language or group of languages which use
same algorithm will have its own slots" — the difference is only that the *grouping key
is the script set*, because that is the only key the encoder can compute reliably. Where
several languages share a script, they share a slot. Grouping *across* scripts is
supported and is a config edit (§3.1.1); it is the right call for scripts that are
individually too small, and the wrong call when done purely by algorithm.

---

## 3. Slot model

### 3.1 Slots for v1

A slot is declared as **`(name, algorithm, scripts[])`**. That single shape covers every
grouping the design needs.

| Slot | Script(s) | Languages absorbed | Algorithm |
|---|---|---|---|
| `NEUTRAL` | `COMMON` | whitespace, digits, punctuation, emoji, math symbols, arrows | BPE + byte fallback |
| `LATIN` | `LATIN` | English, **Hinglish**, **Kanglish**, French, Spanish, … | BPE + byte fallback |
| `DEVANAGARI` | `DEVANAGARI` | Hindi, Marathi, Nepali, Sanskrit, Konkani | BPE + byte fallback |
| `KANNADA` | `KANNADA` | Kannada | BPE + byte fallback |
| `CODE_MATH` | *(structural)* | source code, LaTeX, math | BPE + byte fallback |
| `CATCHALL` | *(fallback)* | any script not claimed above | 256-entry byte table |

`SPECIAL` is not a slot: control tokens own ids `0..k-1`, before every slot.

`CATCHALL` guarantees that text in an unclaimed script (Tamil, Arabic, CJK, unusual
emoji sequences) is still encoded **losslessly** via bytes, rather than becoming
`<unk>`. It is not a trained slot; it is a pure escape hatch.

### 3.1.1 Grouping scripts into one slot

Because a slot takes a *set* of scripts, several scripts can deliberately share one
vocabulary and one algorithm:

```yaml
- name: INDIC_SHARED
  algorithm: bpe
  scripts: [DEVANAGARI, KANNADA]
```

This is the right call when scripts are individually too small to justify a
dedicated vocabulary — the low-resource case.

**Grouping by algorithm alone is measured and rejected.** The tempting argument is
that since there are only two algorithms there should be only two slots, with one
large shared Unigram vocabulary. That conflates the *algorithm* with the
*vocabulary*: a slot exists to give a script its own rows, and the algorithm is
just a property of the slot. Measured on 10 scripts at equal total vocabulary
(`diagnostics/README.md` §17):

| total vocab | one shared Unigram slot | per-script slots | shared is |
|---|---|---|---|
| 128,000 | 1.6592 | **1.5217** | **+9.0% worse** (6 of 10 scripts lose, worst +33%) |
| 256,000 | 1.6221 | **1.3105** | **+23.8% worse** (all 10 scripts lose, worst +45%) |

Mean held-out tokens/word, lower better. **The penalty grows with vocabulary size
rather than shrinking**, which is the opposite of the intuition that more rows
would relieve the competition.

The mechanism is that sharing a vocabulary shares *competition*, not just
capacity. Different scripts occupy nearly disjoint UTF-8 byte ranges, so most
rows can only ever fire for one script; what decides which script's pieces get
learned is pooled corpus frequency. Scripts whose full-word forms are rarer in the
mixture — Odia, Telugu, Malayalam — lose the most, while the pooled model cannot
even fill its budget (206,760 pieces when asked for 256,000 on a 4.6M-character
pool).

This is also what the closest real-world instance shows: sarvam-1's 68k vocabulary
shared across 10 Indic languages scored 30.6% worse on Kannada than a dedicated 32k
slot, with 3x worse utilization (24.2% vs 79.7%).

**What grouping *is* still good for:** folding NEUTRAL and CODE_MATH into one BPE
slot is reasonable — both are ASCII-heavy, small, and adjacent in character
distribution. That takes the model from 13 slots to 12 without touching any
Unigram vocabulary.

Routing is data, not code: the specs produce a `script -> slot` map, and adding a
language to a slot is a config edit. `tokenizer/slots.py` asserts that the default
specs and the segmenter's map agree, so the two cannot drift.

### 3.2 Id layout

```
  0 .. k-1   special tokens (<pad>, <bos>, <eos>, <unk>, <mask>, ...)
             optionally <|slot_sep|> -- see 3.3
 k ..  N1    NEUTRAL
 N1 ..  N2   LATIN
 N2 ..  N3   DEVANAGARI
 N3 ..  N4   KANNADA
 N4 ..  N5   CODE_MATH
 N5 ..  N6   CATCHALL byte tokens (<0x00> .. <0xFF>)
```

Ids are **contiguous and dense**, starting immediately after the special range. The
actual boundary numbers are not written here: they depend on the trained slot sizes and
are emitted into `slots.v1.json`. `first_invalid_id` is simply the end of the range —
derived, never declared.

### 3.3 There is no sentinel id

The original design reserved `999999`. **That is removed.** It was a magic constant
guarding against a sentinel the design deliberately never emits, and it reserved nothing:
an id only exists if it is allocated.

The replacement has three parts, each doing a real job:

1. **`first_invalid_id = vocab_size`.** The whole bound the decoder needs, derived from
   the layout. Any id outside `[0, vocab_size)` is invalid.
2. **`<|slot_sep|>` — an ordinary special token, if you ever want in-stream markers.**
   Adding it to `specials` costs **exactly one vocabulary row**, at a low id. Emitting
   is opt-in (`emit_slot_markers`); the decoder skips it, so round-trip stays exact even
   with markers on. Default off, because the chosen stream format is markerless.
3. **Embedding padding, computed not magic.** `padded_vocab_size = ceil(vocab_size /
   embed_multiple) * embed_multiple`, default `embed_multiple = 128`. This is the one
   legitimate reason to reserve a tail — tensor-parallel embedding sharding wants a
   multiple of 64/128. Padding rows are allocated by the model, recorded in the manifest,
   and never emitted.

The parameter argument that motivated 999999 still holds and still matters: at
`d_model = 2048` with tied embeddings, a 130k vocab costs **~266M** embedding parameters
and a 1M vocab costs **~2.05B**. The fix is to derive the bound from the layout, not to
declare a big number.

### 3.4 Deferred: per-language slots

Note that v1 already has **no per-language slots**: `LATIN` covers every Latin-script
language and `DEVANAGARI` covers Hindi, Marathi, Nepali and Sanskrit. Splitting those
into per-language slots would require inserting a language-ID step into the encoder. The
risks to accept at that point:

- LID errors route text into a slot the model never saw that token in, which is worse
  than a shared vocab.
- The decoder can no longer route by id alone if LID is non-deterministic — the
  language decision must be recorded somewhere.

Out of scope for v1. Revisit only with data.

---

## 4. Artifacts

Frozen before pretraining begins. A retrain shifts every boundary and invalidates every
checkpoint, so these are versioned and checksummed.

| Artifact | Contents |
|---|---|
| `vocab.v1.json` | merged id → token string, all slots, plus `vocab_size`, `first_invalid_id`, `padded_vocab_size`, `embed_multiple` |
| `slots.v1.json` | per-slot `{name, start, end, algorithm, size}` |
| `slot_specs.v1.json` | the `(name, algorithm, scripts[])` declarations, which is where script→slot routing lives |
| `slot_models/*.json` | the per-slot trained tokenizer artifacts |
| `corpora.v1.yaml` | corpus registry used for training (see §9) |
| `tokenizer_v1.manifest.json` | sha256 of all of the above + a single top-level `tokenizer_version` |

`tokenizer_version` is stamped into every training run. Loading a vocab whose manifest
hash does not match the checkpoint's stamp is a hard error, not a warning.

Any future change — new script slot, bigger vocab, different algorithm — produces
`v2`. Slots are **appended** at the end where possible so that v1 ids keep their
meaning; renumbering is a breaking change.

---

## 5. Phase 1 — Train per-slot tokenizers (offline)

For each slot, train its tokenizer on that slot's corpus only.

- **LATIN / DEVANAGARI / KANNADA** → **Unigram**, with `max_piece_length = 48`.
  **NEUTRAL / CODE_MATH** → **BPE**.

  This reverses an earlier BPE decision, and the reversal is the single most
  instructive result in this document. The spec's *original* choice was Unigram
  for Indic, on the theory that agglutinative scripts reward probabilistic
  segmentation. It was "corrected" to BPE twice, and both corrections measured a
  configuration artefact and reported an algorithmic result:

  1. **Synthetic corpus, 75 distinct Devanagari runs** → "Unigram is 3-7x worse".
     Far too small and repetitive for Unigram's EM.
  2. **Real corpus, HF defaults** → "HF Unigram is 15.9% worse than
     SentencePiece". Cause: `max_piece_length` **counts pre-tokenized tokens, and
     under ByteLevel that is a byte budget**, not a character budget. The default
     of 16 caps an Indic piece at ~5 aksharas, where SentencePiece's limit of 16
     counts *characters*. The median Kannada training run is 88 byte-chars.

  With the limit raised, measured on 12 Indic scripts plus romanized code-mixes
  (`diagnostics/README.md` §13-14):

  | case | languages | Unigram margin over BPE |
  |---|---|---|
  | **One slot per SCRIPT, shared** (the design) | Hindi + Marathi + Nepali | **+5.5% / +7.8% / +7.1%** |
  | **LATIN slot, code-mixed** | English / Hinglish / Tanglish | **+8.0% / +9.7% / +5.3%** |
  | One vocabulary per language, 16k | 12 languages | 6 wins, 6 losses, all ≤7% |

  The decisive case is the shared slot, because that is how this design works.
  Further decomposition (`diagnostics/README.md` §15) explains *why*, and the
  reason is not "Unigram is better at these languages":

  | case | vocabulary | Unigram vs BPE |
  |---|---|---|
  | English alone, 32k | single-language | **tie (0.0%)** |
  | English alone, 16k | single-language | +2.0% |
  | 12 Indic languages, one vocab each, 16k | single-language | 6 wins / 6 losses, all ≤7% |
  | Hindi / Kannada alone, 32k | single-language | +6.6% / +7.5% |
  | Devanagari shared (hi+mr+ne) | **shared** | **+5.5% / +7.8% / +7.1%** |
  | LATIN shared (en+hinglish+tanglish) | **shared** | **+8.0% / +9.7% / +5.3%** |

  The operative finding is:

  > **Unigram's advantage is robustness to a shared vocabulary.** For a
  > single-language vocabulary the two are close to interchangeable; as soon as a
  > slot holds more than one language, BPE degrades several times faster.

  Measured cost of sharing: BPE loses **4.5-11.6%** on Devanagari languages and
  **7.6-9.4%** on English, while Unigram loses **0-2.6%** — and for Hindi it
  *gains* (+1.6%), since it benefits from the extra characters where BPE's greedy
  merges merely dilute. BPE is roughly 3-5x more fragile to sharing.

  Since every slot here is either shared by construction, unavoidably shared
  (LATIN cannot separate romanized Indic from English), or morphologically rich,
  Unigram is the right default. **The choice would flip for a genuinely
  monolingual, analytic-language slot** — worth knowing if this design is ever
  reused for, say, an English-only model.

  Caveat: Hindi measured +6.6% for Unigram at 32k/5M chars but −0.4% at
  16k/1.5M. Both budget and data size changed together, so they cannot be
  separated here; the advantage appears to grow with both. Unigram training is
  also **~10x slower** (50-135s vs 3-9s per slot).

  On providers: only HF `tokenizers` and SentencePiece are maintained *and* able
  to train (`tiktoken` is inference-only; `youtokentome`, `subword-nmt`,
  `fastbpe` are abandoned). HF BPE and SentencePiece BPE agree within 2% at every
  budget — the BPE implementations were never in dispute.

  Separately, SuperBPE (COLM 2025) independently identifies **whitespace
  ownership** — not the merge algorithm — as the dominant tokenizer variable,
  reporting up to 33% fewer tokens from allowing pieces to span words. That is
  the same effect as this spec's `leading` mode, and good corroboration that §5.1
  was the right thing to focus on.
- **LATIN / NEUTRAL / CODE_MATH** → **BPE**.
- **`byte_fallback = True` on every slot.** Not optional. This is what makes the
  round-trip invariant (§8) achievable and drives the unknown rate to zero.
- **`CATCHALL`** is not trained — it is a fixed 256-token byte table, though it can be
  seeded with the 256 byte tokens shared with the other slots.

### 5.1 Whitespace ownership — a measured fork

This is the fiddliest correctness issue in the design. If spaces live inside the LATIN
run, they get encoded with `Ġ`-style markers; if they live in the NEUTRAL slot, they are
their own runs. Library conventions differ (`▁` U+2581 for SentencePiece, `Ġ` U+0120 for
GPT-2-style BPE), and a mismatch here produces whitespace drift that is easy to miss.

Two variants, both implemented and compared:

- **`whitespace_ownership: neutral` (default)** — whitespace is stripped from every
  script corpus and becomes its own NEUTRAL run. Renders run boundaries unambiguous and
  makes exact round-trip trivial, because no tokenizer ever sees a leading space.
- **`whitespace_ownership: leading`** — the sentencepiece `▁` convention: each run keeps
  its leading whitespace and the script tokenizer encodes it. Closer to standard
  SentencePiece/BPE behaviour and may give better LATIN fertility.

**`leading` is the default.** Measurement decided it, across three evaluations —
two of which were invalid and had to be discarded (§12, `diagnostics/README.md` §5).
On a sequence-novel natural held-out set, `neutral` lost to the shared-vocab
baseline on Devanagari (−61.9%) and Latin (−65.9%), while `leading` beat it on
every script (Devanagari +24.4%, Kannada +47.6%, Latin +18.6%, code/math +94.1%).
`neutral` costs a token per space and forbids cross-word merges; `leading` removes
both. Set `whitespace_ownership: neutral` only to reproduce the old numbers.

### 5.2 Normalization

- Unicode **NFC** by default. NFKC is *not* used: it is not invertible and it folds
  Indic letterforms in ways that are lossy and linguistically wrong for Devanagari and
  Kannada.
- **ZWJ (U+200D) and ZWNJ (U+200C) are preserved.** They are semantically load-bearing
  in Indic conjunct formation. Stripping them silently corrupts text.
- Normalization is recorded in the manifest. The round-trip invariant is defined against
  *normalized* text (§8) since normalization is by design not invertible.

---

## 6. Phase 2 — Merge

```
final_vocab = SPECIAL ++ NEUTRAL ++ LATIN ++ DEVANAGARI ++ KANNADA ++ CODE_MATH ++ CATCHALL
```

Id assignment walks the slots in that fixed order. Each slot's local id 0 becomes the
slot's global `start`.

### 6.1 Budget allocation

You chose **proportional to corpus size per script**. Pure proportionality (`alpha = 1.0`)
starves the smaller scripts, because Latin-script corpora dominate raw byte counts — the
exact failure this tokenizer exists to prevent. Allocation is therefore:

```
budget_s = clamp( round( (bytes_s ^ alpha) / Σ_j (bytes_j ^ alpha) * TOTAL ),  MIN_SLOT, MAX_SLOT )

alpha      = 0.7    # 1.0 = literal proportional; 0.5 = squareroot-tempered
MIN_SLOT   = 2000   # floor so no script is starved
MAX_SLOT   = 64000  # ceiling so no script hogs the embedding table
TOTAL      = configured total vocab
```

`alpha = 0.7` is the draft default. **Set `alpha = 1.0` at review if you want the literal
proportional rule** — the tradeoff is that small scripts get floor-allocated only.

NEUTRAL and CATCHALL are exempt from this formula (NEUTRAL is sized by its own corpus;
CATCHALL is fixed at 256).

### 6.1.1 Recommended configuration (measured)

Algorithm and piece-limit choices below are **measured**; vocabulary sizes are
**recommended defaults** calibrated against measured relative difficulty, because
fertility is monotonically decreasing in vocabulary size and therefore has no
interior optimum to find. See `diagnostics/README.md` §16.

Difficulty is held-out tokens/word at a common 32k budget, so it is directly
comparable across scripts — higher means the script needs more rows.

| Slot | Script | Languages routed here | Algorithm | `max_piece_length` | **Vocab** | measured cost of this size |
|---|---|---|---|---|---|---|
| `NEUTRAL` | Common | whitespace, digits, punctuation, emoji, math symbols | BPE | – | **1,024** | – |
| `LATIN` | Latin | English, **Hinglish, Kanglish, Tenglish, Tanglish, Manglish** | **Unigram** | 48 | **64,000** | ref: 1.2875 t/w |
| `DEVANAGARI` | Devanagari | Hindi, Marathi | **Unigram** | 48 | **48,000** | ref: 1.2291 t/w |
| `KANNADA` | Kannada | Kannada | **Unigram** | 48 | **32,000** | **+5.8%** vs 48k |
| `MALAYALAM` | Malayalam | Malayalam | **Unigram** | 48 | **32,000** | **+9.1%** vs 48k |
| `TELUGU` | Telugu | Telugu | **Unigram** | 48 | **32,000** | **+9.6%** vs 48k |
| `TAMIL` | Tamil | Tamil | **Unigram** | 48 | **32,000** | **+3.9%** vs 40k |
| `BENGALI` | Bengali | Bengali, **Assamese** | **Unigram** | 48 | **32,000** | ref: 1.0089 t/w |
| `GUJARATI` | Gujarati | Gujarati | **Unigram** | 48 | **24,000** | ref: 1.1002 t/w |
| `ORIYA` | Oriya | Odia | **Unigram** | 48 | **16,000** | **+8.5%** vs 24k |
| `GURMUKHI` | Gurmukhi | Punjabi | **Unigram** | 48 | **16,000** | **+5.5%** vs 20k |
| `CODE_MATH` | structural | source code, LaTeX, math | BPE | – | **16,000** | – |
| `CATCHALL` | fallback | unmodelled scripts | bytes | – | **256** | – |
| | | | | | **≈ 345,280 total** | |

Costs are measured held-out tokens/word at the reduced size versus the previous
one, same runs, `diagnostics/README.md` §18. Utilization improves at every
reduced slot (e.g. Kannada 0.316 → 0.391), which is expected when a vocabulary is
shrunk.

**Budget history:**

| version | total | change |
|---|---|---|
| v1 (14 slots) | 449,280 | first full allocation |
| v2 (13 slots) | 413,280 | −8.0%: dropped `ARABIC`, right-sized `DEVANAGARI` |
| **v3 (13 slots)** | **345,280** | **−16.5%**: trimmed the Dravidian and minor-script slots |

Total embedding parameters at `d_model=2048` tied: **~707M** (v1 was ~920M).

**Two things to note about the v3 trim:**

1. **The two largest costs land on the two hardest scripts.** Telugu (+9.6%) and
   Malayalam (+9.1%) were already the scripts with the highest fertility, so the
   trim costs most where the model can least afford it. If any slot is worth
   keeping larger, it is these two.
2. **Assamese has no separate budget to reduce.** It shares the Bengali script,
   so it lives inside `BENGALI` at 32k. Trimming "Assamese" would mean trimming
   Bengali; if that is wanted, 24k is the next step and would cost Bengali
   roughly 8% (about +1.09 t/w on the same curve). Otherwise Assamese rides along
   free, which is the usual reason for sharing a script slot in the first place.

Sizing rationale, in order of weight:

1. **Shared slots get the most.** `LATIN` and `DEVANAGARI` each carry several
   languages and both showed the largest Unigram-over-BPE margins (+8.0-9.7% and
   +5.5-7.8%), so they earn the ceiling.
2. **Harder scripts get more rows.** Malayalam, Kannada and Telugu still sit at
   1.5-1.75 tokens/word at 32k while Urdu and Punjabi are already below 0.9.
3. **`NEUTRAL` is a small closed set** — whitespace, digits, punctuation, emoji —
   and needs almost nothing.
4. **`CODE_MATH` stays BPE**, the conventional choice, untested here.

**Scaling.** If ≈449k is too large, halving any slot costs roughly **9-16%
fertility** for that script (measured marginal gain per doubling, §16). The
cheapest place to cut is the low-difficulty slots: `GURMUKHI`, `ARABIC`, `ORIYA`
and `GUJARATI` at 12-16k cost little, since they already sit below 1.05
tokens/word.

**Caveats.** (a) Sizes are calibrated on ~1.3M-character training corpora, which
cannot even fill a 64k vocabulary — the trainers capped at 62-68k when asked for
100k. Real corpora will support larger slots, so treat these as conservative.
(b) Vocabulary utilization measured 0.21-0.81 on small held-out sets; the real
figure is higher, since a piece that fires once in 100M tokens is healthy and
will not appear in 60k words.

### 6.2 Duplicate strings across slots

`"the"` may exist in both LATIN and CODE_MATH. **Keep both.** Deduplicating into a
shared id would break the invariant that a slot lookup determines the algorithm, which
is the entire mechanism of §7.

Cost: a small number of wasted ids plus separate embedding rows. Accepted knowingly.

### 6.3 Assertions at merge time

1. No duplicate id ranges; slots are contiguous and ordered, starting at `special_end`.
2. No script is claimed by two slots; exactly one `bytes` slot exists.
3. `first_invalid_id == vocab_size`; every non-special id routes to exactly one slot.
4. No duplicate token *string* within a single slot.
5. Each special token appears exactly once, in the special range.
6. `padded_vocab_size >= vocab_size` and is a multiple of `embed_multiple`.

---

## 7. Phase 3 — Encode

```
raw -> normalize(NFC) -> segment into script runs -> route each run to its slot -> emit ids in ORIGINAL order
```

1. **Normalize** per §5.2.
2. **Segment** into maximal runs of one class. Classification is deterministic:
   - Unicode script property → `LATIN`, `DEVANAGARI`, `KANNADA`
   - Unicode category `Z*`, `Nd`, `P*`, `S*` (non-letter symbols), emoji → `NEUTRAL`
   - unlisted scripts → `CATCHALL`
   - **structural override**: fenced code blocks (```), inline code, and math delimiters
     (`$…$`, `$$…$$`, `\(…\)`) override the script classification and route to
     `CODE_MATH`. This is the "keep code and math separate" rule from the original design,
     made deterministic by using *structure* rather than a classifier.
   - Documents whose source is known to be code (a whole file from a code corpus) are
     routed to `CODE_MATH` wholesale, without per-run segmentation.
3. **Route**: each run goes to its slot's tokenizer.
4. **Emit** the ids in original order, concatenated. No sentinels, no tags.

### 7.1 Invariants

- Every emitted id belongs to exactly one slot.
- `max(id) < first_invalid_id`.
- Order is preserved: the i-th run's ids occupy a contiguous span, and spans appear in
  input order.

---

## 8. Phase 4 — Decode

1. Load `slots.v1.json`; build a sorted boundary array.
2. For each id, **binary-search** the boundary array → slot.
3. Group *consecutive* ids by slot, preserving order.
4. Decode each group with **that slot's** algorithm.
5. Concatenate the resulting strings.

Because routing is by id range only, the decoder never needs language identification and
never needs a sentinel. This is the payoff of the partitioned id space.

### 8.1 Round-trip invariant (hard gate)

```
decode(encode(x)) == normalize(x)      for every x in the corpus
```

This must hold **exactly, for 100% of samples**, not 99%. Whitespace drift at run
boundaries is the expected failure mode and the reason §5.1 is a measured fork rather
than an assumption. Byte fallback plus lossless-per-run behavior is what makes this
achievable: if every slot's tokenizer round-trips its own run exactly, concatenation is
exact by construction.

---

## 9. Corpus registry

Config-driven, so the synthetic generator and real corpora are interchangeable:

```yaml
# corpora.v1.yaml
seed: 1337
corpora:
  - name: indic-corpus-hi
    slot: DEVANAGARI
    paths: ["<PATHS FROM YOU>"]
    glob: "**/*.txt"
    encoding: utf-8
    weight: 1.0
  - name: code-stack
    slot: CODE_MATH
    paths: ["<PATHS FROM YOU>"]
    glob: "**/*.py"
    weight: 1.0
  - name: synthetic-demo
    slot: LATIN
    generator: synthetic_v1
    size: 10000        # words; for pipeline smoke-testing only
    weight: 0.0        # weight 0 = present but not trained on
```

**Blocking input needed: the real corpus paths.** The synthetic generator is retained
only as a smoke-test corpus for CI and for proving the round-trip gate — it is far too
small to train a usable vocabulary and is not a training source.

---

## 10. Baseline and acceptance gates

### 10.1 Baseline

One SentencePiece Unigram model over a script-balanced mix of the *same* corpora, sized
to the **same total vocab** as the partitioned version. Equal embedding-parameter count
on both sides, so the comparison is about vocabulary structure and nothing else.

### 10.2 Metrics

| Metric | Why |
|---|---|
| **bits-per-byte (bpb)** on held-out per-script text | **The decider.** Comparable across tokenizers with different vocab sizes, unlike per-token loss. |
| tokens/word (fertility) per script | Directly multiplies pretraining FLOPs |
| bytes/token (compression) | Corpus throughput |
| round-trip exactness | Gate G1 |
| unknown rate | Gate G2 — must be 0 |
| vocab utilization (% ids used ≥ N times) | Detects a bloated or dead slot |
| embedding parameter count | Budget |

### 10.3 Gates

| Gate | Requirement |
|---|---|
| G1 | `decode(encode(x)) == normalize(x)` for 100% of corpus samples |
| G2 | unknown rate = 0 across all scripts |
| G3 | id space sound: every non-special id routes to exactly one slot; `first_invalid_id == vocab_size` |
| G4 | Devanagari and Kannada fertility strictly better than the baseline |
| G5 | No script's bpb worse than baseline by more than 1% |
| G6 | Manifest hashes match between vocab and checkpoint stamp |
| G7 | Whitespace-ownership variant chosen by measurement, not by assertion |

G4 + G5 together are the actual thesis of this design: **the partitioned vocab must beat
the shared vocab where it claims to, and must not lose anywhere else.** If G4 fails, the
design does not do the thing it exists to do and we revisit.

---

## 11. Open items

| # | Item | Owner |
|---|---|---|
| O1 | **Real corpus paths** — required before Phase 1 | you |
| O2 | `alpha`: 0.7 (tempered, draft default) vs 1.0 (literal proportional) | you, at review |
| ~~O3~~ | ~~slot groupings~~ — **resolved**: slots are `(name, algorithm, scripts[])` and grouping is config (`config/slots.v1.yaml`); no per-language slots exist | — |
| ~~O4~~ | ~~`whitespace_ownership` default~~ — **resolved**: `leading` is the default, decided by measurement across three evaluations | — |
| O5 | Confirm `CODE_MATH` detection by structure + source, not by classifier | you, at review |
| O6 | Total vocab `TOTAL` for v1 | you |
| O7 | Confirm the `CATCHALL` escape hatch is acceptable vs erroring on unknown scripts | you |
| O8 | Which scripts to fold into a shared low-resource slot (e.g. `INDIC_SHARED`), and which to leave on byte fallback | you |
| O9 | `embed_multiple`: 128 default for tensor-parallel sharding — confirm your target hardware | you |

---

## 12. Implementation status and measured findings

### 12.1 Built and verified

The pipeline in §5–§8 is implemented in `tokenizer/` and runs end to end:

```
python -m tokenizer.build --out artifacts/v1 --with-baseline
```

| Gate | Status | Evidence |
|---|---|---|
| G1 round-trip exact | **PASS** | 557/557 corpus documents, 43/43 adversarial cases |
| G2 zero unknown | **PASS** | 0 special ids and no `<unk>` emitted; escape hatch is CATCHALL |
| G3 id space sound | **PASS** | every non-special id routes to exactly one slot; `first_invalid_id == vocab_size` |
| G4 Indic fertility better | **PASS at natural budget**, FAIL at matched | held-out: Devanagari +24.4%, Kannada +47.6% at vocab 5938; Devanagari −43.9% at vocab 1500 |
| G5 no script regression | **PASS at natural budget**, FAIL at matched | held-out: all four scripts improve at vocab 5938; Latin −99.7% at vocab 1500 |
| G6 manifest integrity | **PASS** | hashes verified; tamper detection covered by test |

176 tests pass (`python -m pytest`), including 12 that verify the bits-per-byte
math against closed-form values. Adversarial coverage was widened from 37 to 43
cases to include Tamil, Telugu, Malayalam, Bengali, Hebrew, Hangul and Cyrillic.

The decider named in §10.2 is now measured — see §12.5. `python -m tokenizer.bpb`
trains a small transformer on each tokenizer's output for an identical token budget
and reports held-out bits-per-byte.

Current build: `vocab_size=5938`, `padded_vocab_size=6016`, `first_invalid_id=5938`,
`whitespace_ownership=leading`, `alphabet_scope=observed`. The grouped variant in
`config/slots.grouped.example.yaml` folds Devanagari and Kannada into one slot.

### 12.2 Findings that changed the design

1. **Unigram was the wrong choice for Indic.** Replaced with BPE. This was an
   unvalidated assertion in the original spec and was the largest error in it.
   Full evidence in `diagnostics/README.md`.
2. **Whitespace ownership is a first-order cost, not a detail.** Replacing the
   `neutral` default with `leading` turned Devanagari from −61.9% into +24.4%
   against the baseline and Latin from −65.9% into +18.6%. Getting this measurement
   right took three attempts, two of which were invalid (memorisation, then an
   over-corrected word-shuffled held-out biased toward word-level models). The
   history is in `diagnostics/README.md` §5 and is the strongest argument in this
   document for never trusting a single evaluation.
3. **Every slot paid a 256-row byte-alphabet tax.** Seeding all 256 byte characters
   into each slot meant that at a 1500-row budget, 88.7% of the vocabulary was raw
   bytes and only ~38 rows per slot were learned subwords. Seeding only the bytes
   each slot's corpus actually uses (40–59 of them) reclaimed ~1311 rows, taking
   learned rows from ~190 to 1533. See `diagnostics/README.md` §4.
4. **`unk_token=None` silently destroys data.** HF's BPE *discards* out-of-alphabet
   characters without emitting any id when `unk_token` is None, so `"\n\n\t"`
   decoded to `"\n\n"` with nothing in the id stream to detect it. The full 256-byte
   seed had masked this from the start. Fixed by setting `unk_token="<unk>"`.
5. **Losslessness is achieved by construction, not by model quality.**
   `ByteLevel(add_prefix_space=False, use_regex=False)` plus the shared CATCHALL
   fallback makes every slot a lossless codec. This is why G1/G2 hold across Tamil,
   Arabic, CJK, Greek, emoji ZWJ sequences, control characters, ligatures and
   ZWJ/ZWNJ conjuncts.
6. **Unmodelled scripts never need a trained model.** CATCHALL is a pure byte
   table: Tamil encodes as 15 ids for 5 characters, exactly the UTF-8 byte count,
   and round-trips.
7. **The `999999` sentinel was removed.** It guarded nothing and reserved nothing.
   Replaced by a derived bound (`first_invalid_id`), an optional one-row
   `<|slot_sep|>` special, and computed embedding padding. See §3.3.
8. **Slot grouping became config, not code.** A slot is `(name, algorithm,
   scripts[])`, so multi-script grouping is a config edit. Script detection was
   expanded from 3 scripts to 27, table-driven, with everything unclaimed falling
   through to lossless byte fallback.
9. **A runaway structural span produced a confident, wrong verdict.** Allowing the
   fence pattern to match to end-of-input let one unterminated delimiter route the
   remainder of a concatenated corpus into `CODE_MATH`, which byte-fell-back and
   inflated tokens ~4.5x. It made the partitioned tokenizer look decisively worse
   on bits-per-byte (`-0.41`) when it is in fact better (`+0.009`). Delimiters must
   now close, and the harness encodes document by document. See
   `diagnostics/README.md` §6.
10. **Fallback granularity is a design property, not a detail.** Firing fallback
    for the whole run meant one character outside a slot's alphabet re-encoded
    everything containing it: a 2801-character Devanagari run cost 7601 tokens
    instead of 2400 because of a single leading newline. Fallback now isolates the
    offending characters by bisection (`O(k log n)`), so the same case costs 2420
    tokens with 3 fallback characters. See `diagnostics/README.md` §7.

### 12.3 Why the smoke-corpus numbers are still not evidence

The generator is now v2 (combinatorial: 128M possible English sentences, 812k Hindi,
423k Kannada, 58k Hinglish, 41k Kanglish), so the memorisation that invalidated the
earlier measurements is largely gone — the held-out **memorisation gap** is now
reported alongside every comparison for exactly this reason, and the residual
sentence overlap between splits is 0.197, driven mainly by the small Hinglish and
Kanglish spaces.

It is still a synthetic corpus, and every number above must be re-measured on real
corpora (O1) before it means anything about real performance. The evaluation
history in `diagnostics/README.md` §5 is a concrete warning: two of three prior
evaluations produced a confident, wrong answer.

### 12.4 What remains

- **Re-run everything on real corpora (O1).** This is now the only substantive gap.
  The protocol, the harness and its correctness tests all exist; only the data does
  not. Every margin in §12.5 is fragile because the corpus is synthetic.
- Decide which scripts to fold into a shared low-resource slot (O8). The machinery
  is built and tested; the choice needs real corpus sizes.
- Re-check the Unigram-vs-BPE decision under `leading`, since `leading` produces
  longer runs and Unigram previously improved on longer sequences.
- If the capacity cost turns out to matter, the lever is the per-slot byte seed:
  `observed+ascii` still spends ~141 rows on a Devanagari slot whose runs can only
  ever contain Devanagari bytes plus whitespace. Slots other than LATIN plausibly
  need only ~50 rows.

### 12.5 Bits per byte: the decider, measured

A small transformer (4 layers, `d_model=128`, tied embeddings, 819,200 training
tokens) trained on each tokenizer's output and scored on sequence-novel held-out
text. Full table and method in `diagnostics/README.md` §8.

| Budget | partitioned vocab | baseline vocab | params | **bpb delta** |
|---|---|---|---|---|
| natural | 7281 | 7281 (BPE) | equal | **+0.0257** |
| natural | 7291 | 7285 (BPE) | equal | **+0.0173** |
| natural | 7291 | 5392 (Unigram, pruned) | +16% | +0.0092 |
| matched | 1544 | 1544 (BPE) | equal | **−0.1605** |
| matched | 1544 | 1544 (Unigram) | equal | −0.0844 |

Positive means the partitioned tokenizer wins. **Each sign was reproduced in an
independent process.**

**The thesis survives, with a condition attached.** At its natural vocabulary and
equal parameter count the partitioned tokenizer produces better bits-per-byte. At
matched vocabulary it loses badly. The design is capacity-hungry: it needs roughly
4.7x the vocabulary of a shared model, because six slots each pay a byte seed plus
their own subwords while a shared vocabulary concentrates its whole budget.

G4 and G5 in §10.3 should therefore be read as *conditional*: they hold at the
natural budget and fail at matched budget, and the honest statement of the design's
claim is "better bits-per-byte per parameter, at a larger vocabulary".

One result worth keeping: at equal parameter count the BPE baseline emits **fewer**
tokens per byte (0.0655 vs 0.0731) and still has worse bpb. The natural-budget win
is about the quality of the units, not about token count — and the earlier
"1.9x more token-efficient" claim (row 3) was an artifact of a Unigram baseline
that had pruned itself to 5,392 of 7,291 requested rows.

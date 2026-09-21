"""Corpus registry and the synthetic smoke-test generator.

The synthetic corpus is NOT a training source. It exists to exercise the pipeline
and to give the tokenizer comparison something to measure on.

Bumped to generator v2 for a specific reason: v1 drew from ~20 fixed sentences
per language, so a slot with a few hundred dedicated tokens could memorise every
sentence it ever saw and fertility became a memorisation score. v2 composes
sentences combinatorially from word banks and grammar templates, so the sentence
space is enormous and two different seeds yield largely disjoint, still-natural
text. That makes a genuine held-out split possible.
"""

from __future__ import annotations

import glob as globlib
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .scripts import CODE_MATH, DEFAULT_WHITESPACE_OWNERSHIP, segment

# --------------------------------------------------------------------------
# Word banks
# --------------------------------------------------------------------------

# --- English ---------------------------------------------------------------
_EN = {
    "det": ["the", "a", "my", "his", "her", "their", "this", "that", "our"],
    "adj": ["quick", "lazy", "bright", "quiet", "old", "new", "small", "large",
            "happy", "careful", "young", "tall", "clever", "gentle"],
    "noun": ["teacher", "student", "child", "brother", "sister", "doctor",
             "farmer", "engineer", "artist", "writer", "village", "river",
             "garden", "machine", "letter", "story", "picture", "market",
             "school", "library", "bridge", "mountain", "song", "problem"],
    "verb": ["reads", "writes", "sees", "builds", "repairs", "washes",
             "carries", "opens", "closes", "finds", "draws", "sells",
             "paints", "studies"],
    "prep": ["near", "behind", "beside", "above", "below", "inside", "around"],
    "adv": ["quickly", "slowly", "carefully", "quietly", "daily", "often",
            "patiently", "eagerly"],
}

_EN_TEMPLATES = [
    "{det} {adj} {noun} {verb} {det2} {noun2}.",
    "{det} {noun} {adv} {verb} {det2} {noun2}.",
    "{det} {noun} {verb} {prep} {det2} {adj2} {noun2}.",
    "{det} {adj} {noun} {adv} {verb}.",
    "{det} {noun} and {det2} {noun2} {verb} {adv}.",
]

# --- Hindi (Devanagari) ----------------------------------------------------
# Subjects carry gender/number so the verb can agree, which keeps the corpus
# linguistically plausible rather than word salad.
_HI_SUBJ = {
    "m": ["मैं", "तुम", "वह", "राम", "बच्चा", "शिक्षक", "विद्यार्थी", "पिता",
          "भाई", "मित्र", "किसान", "डॉक्टर", "लेखक"],
    "f": ["सीता", "माँ", "बहन", "लड़की", "शिक्षिका", "कवयित्री", "दादी"],
    "p": ["हम", "वे", "बच्चे", "लोग", "शिक्षक", "किसान", "मित्र"],
}
_HI_VERB = {
    "m": ["पढ़ता है", "लिखता है", "पीता है", "खाता है", "गाता है", "देखता है",
          "सुनता है", "बनाता है", "भेजता है", "खरीदता है", "धोता है"],
    "f": ["पढ़ती है", "लिखती है", "पीती है", "खाती है", "गाती है", "देखती है",
          "सुनती है", "बनाती है", "भेजती है", "खरीदती है", "धोती है"],
    "p": ["पढ़ते हैं", "लिखते हैं", "पीते हैं", "खाते हैं", "गाते हैं",
          "देखते हैं", "सुनते हैं", "बनाते हैं", "भेजते हैं", "खरीदते हैं"],
}
_HI_OBJ = ["पुस्तक", "पानी", "खाना", "चाय", "पत्र", "कविता", "चित्र", "गाना",
           "कहानी", "समाचार", "फूल", "फल", "दूध", "अखबार", "कपड़े", "सब्ज़ी",
           "दवा", "रोटी", "मिठाई", "बगीचा"]
_HI_ADJ = ["अच्छा", "सुंदर", "नया", "पुराना", "बड़ा", "छोटा", "मीठा", "ठंडा",
           "गरम", "तेज़", "साफ़", "महँगा", "सस्ता", "लंबा"]
_HI_ADV = ["धीरे", "जल्दी", "रोज़", "कभी-कभी", "ध्यान से", "अच्छे से", "बहुत",
           "फिर", "आज", "कल"]

_HI_TEMPLATES = [
    "{s} {o} {v}।",
    "{s} {adj} {o} {v}।",
    "{s} {adv} {o} {v}।",
    "{s} {o} {adv} {v}।",
    "{s} और {s2} {o} {v}।",
]

# --- Kannada ---------------------------------------------------------------
_KN_SUBJ = {
    "1": ["ನಾನು"],
    "2": ["ನೀನು"],
    "m": ["ಅವನು", "ರಾಮ", "ಶಿಕ್ಷಕ", "ವಿದ್ಯಾರ್ಥಿ", "ಅಣ್ಣ", "ತಂದೆ", "ರೈತ",
          "ವೈದ್ಯ", "ಬರಹಗಾರ"],
    "f": ["ಅವಳು", "ಸೀತಾ", "ಅಕ್ಕ", "ತಾಯಿ", "ಶಿಕ್ಷಕಿ", "ಹುಡುಗಿ", "ಅಜ್ಜಿ"],
    "p": ["ಅವರು", "ನಾವು", "ಮಕ್ಕಳು", "ಜನರು", "ಸ್ನೇಹಿತರು", "ರೈತರು"],
}
_KN_VERB = {
    "1": ["ಓದುತ್ತೇನೆ", "ಬರೆಯುತ್ತೇನೆ", "ಕುಡಿಯುತ್ತೇನೆ", "ತಿನ್ನುತ್ತೇನೆ",
          "ನೋಡುತ್ತೇನೆ", "ಕೇಳುತ್ತೇನೆ", "ಮಾಡುತ್ತೇನೆ"],
    "2": ["ಓದುತ್ತೀಯೆ", "ಬರೆಯುತ್ತೀಯೆ", "ಕುಡಿಯುತ್ತೀಯೆ", "ತಿನ್ನುತ್ತೀಯೆ",
          "ನೋಡುತ್ತೀಯೆ", "ಕೇಳುತ್ತೀಯೆ", "ಮಾಡುತ್ತೀಯೆ"],
    "m": ["ಓದುತ್ತಾನೆ", "ಬರೆಯುತ್ತಾನೆ", "ಕುಡಿಯುತ್ತಾನೆ", "ತಿನ್ನುತ್ತಾನೆ",
          "ನೋಡುತ್ತಾನೆ", "ಕೇಳುತ್ತಾನೆ", "ಮಾಡುತ್ತಾನೆ"],
    "f": ["ಓದುತ್ತಾಳೆ", "ಬರೆಯುತ್ತಾಳೆ", "ಕುಡಿಯುತ್ತಾಳೆ", "ತಿನ್ನುತ್ತಾಳೆ",
          "ನೋಡುತ್ತಾಳೆ", "ಕೇಳುತ್ತಾಳೆ", "ಮಾಡುತ್ತಾಳೆ"],
    "p": ["ಓದುತ್ತಾರೆ", "ಬರೆಯುತ್ತಾರೆ", "ಕುಡಿಯುತ್ತಾರೆ", "ತಿನ್ನುತ್ತಾರೆ",
          "ನೋಡುತ್ತಾರೆ", "ಕೇಳುತ್ತಾರೆ", "ಮಾಡುತ್ತಾರೆ"],
}
_KN_OBJ = ["ಪುಸ್ತಕ", "ನೀರು", "ಊಟ", "ಚಹಾ", "ಪತ್ರ", "ಕವನ", "ಚಿತ್ರ", "ಹಾಡು",
           "ಕಥೆ", "ಸುದ್ದಿ", "ಹೂವು", "ಹಣ್ಣು", "ಹಾಲು", "ಪತ್ರಿಕೆ", "ಬಟ್ಟೆ",
           "ತರಕಾರಿ", "ಔಷಧ", "ರೊಟ್ಟಿ", "ಸಿಹಿ", "ಉದ್ಯಾನ"]
_KN_ADJ = ["ಒಳ್ಳೆಯ", "ಸುಂದರ", "ಹೊಸ", "ಹಳೆಯ", "ದೊಡ್ಡ", "ಚಿಕ್ಕ", "ಸಿಹಿ",
           "ತಣ್ಣನೆ", "ಬಿಸಿ", "ವೇಗ", "ಸ್ವಚ್ಛ", "ದುಬಾರಿ", "ಅಗ್ಗ", "ಉದ್ದ"]
_KN_ADV = ["ನಿಧಾನವಾಗಿ", "ಬೇಗ", "ಪ್ರತಿದಿನ", "ಎಚ್ಚರಿಕೆಯಿಂದ", "ಚೆನ್ನಾಗಿ",
           "ಬಹಳ", "ಮತ್ತೆ", "ಇಂದು", "ನಾಳೆ"]

_KN_TEMPLATES = [
    "{s} {o} {v}.",
    "{s} {adj} {o} {v}.",
    "{s} {adv} {o} {v}.",
    "{s} {o} {adv} {v}.",
    "{s} ಮತ್ತು {s2} {o} {v}.",
]

# --- Hinglish (romanised Hindi + English) ----------------------------------
_HINGLISH_TEMPLATES = [
    "yaar {obj} bahut {adj} hai",
    "{pron} {obj} {vinf} raha hai",
    "mujhe {obj} {vinf} hai",
    "{pron} ne {obj} {vpast}",
    "please {obj} {vinf} do",
    "{pron} ko {obj} {adj} lagta hai",
    "kal {pron} {obj} {vpast}",
]
_HI_ROM_PRON = ["main", "tum", "wo", "hum", "ye", "log"]
_HI_ROM_OBJ = ["khana", "paani", "chai", "kitaab", "kaam", "gaana", "photo",
               "movie", "letter", "news", "bijli", "paisa"]
_HI_ROM_ADJ = ["accha", "sundar", "naya", "purana", "bada", "chhota", "meetha",
               "thanda", "garam", "zaroori"]
_HI_ROM_VINF = ["khaana", "peena", "padhna", "likhna", "dekhna", "sunna",
                "banaana", "bhejna", "kharidna"]
_HI_ROM_VPAST = ["khaaya", "piya", "padha", "likha", "dekha", "suna",
                 "banaaya", "bheja", "kharida"]

# --- Kanglish (romanised Kannada + English) --------------------------------
_KANGLISH_TEMPLATES = [
    "nanu {obj} {vinf} beku",
    "{pron} {obj} {vpast}",
    "idu tumba {adj} ide",
    "{pron} {obj} nodi",
    "naale {pron} {obj} {vinf}",
]
_KN_ROM_PRON = ["nanu", "ninu", "avanu", "avalu", "avaru", "naavu"]
_KN_ROM_OBJ = ["oota", "neeru", "chaha", "pustaka", "kelasa", "haadu", "photo",
               "cinema", "patra", "suddi", "vidyut", "hana"]
_KN_ROM_ADJ = ["chennagide", "sundara", "hosa", "haleya", "dodda", "chikka",
               "sihi", "tumba", "mukhya"]
_KN_ROM_VINF = ["maadalu", "nodalu", "helalu", "kelalu", "odalu", "bareyalu",
                "kodal", "tinalu"]
_KN_ROM_VPAST = ["maadide", "nodeda", "helida", "kelida", "odida", "bareyide",
                 "kotide", "tindide"]

_CODE_BLOCKS = [
    """```python
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```""",
    """```python
import json

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
```""",
    """```python
class Tokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self._vocab = {}

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError
```""",
]

_MATH_BLOCKS = [
    r"$$\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}$$",
    r"$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$",
    r"$$f(x) = \sigma\left(\sum_{i} w_i x_i + b\right)$$",
    r"$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$",
]

_INLINE_MATH = ["$E = mc^2$", r"$a^2 + b^2 = c^2$", r"$\alpha + \beta = \gamma$"]

_MD_LINES = [
    "# Multilingual Notes",
    "## Introduction",
    "This section covers tokenization.",
    "- first item",
    "- second item with **bold** text",
    "- third item with *italic* text",
    "1. numbered one",
    "2. numbered two",
    "See the [documentation](https://example.com/docs) for details.",
    "> A quoted observation about tokenizers.",
    "",
    "| Language | Script | Slot |",
    "| --- | --- | --- |",
    "| Hindi | Devanagari | DEVANAGARI |",
    "| Kannada | Kannada | KANNADA |",
    "| English | Latin | LATIN |",
]

_MIXED_TEMPLATES = [
    "The function {hi} काम कर रहा है and {kn} to everyone {emoji}",
    "{en} {hi} — that is the whole point {emoji}",
    "yaar {kn} means the same thing, no {emoji}",
    "{hi} {kn} {en}",
]

_EMOJI = ["🚀", "🔥", "✨", "🙏", "😀", "🎉", "📚", "🧮", "🌊", "☀️"]


@dataclass
class Document:
    doc_id: str
    kind: str
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Sentence builders
# --------------------------------------------------------------------------


def _english_sentence(rng: random.Random) -> str:
    t = rng.choice(_EN_TEMPLATES)
    return t.format(
        det=rng.choice(_EN["det"]),
        det2=rng.choice(_EN["det"]),
        adj=rng.choice(_EN["adj"]),
        adj2=rng.choice(_EN["adj"]),
        noun=rng.choice(_EN["noun"]),
        noun2=rng.choice(_EN["noun"]),
        verb=rng.choice(_EN["verb"]),
        prep=rng.choice(_EN["prep"]),
        adv=rng.choice(_EN["adv"]),
    )


def _hindi_sentence(rng: random.Random) -> str:
    g = rng.choice(("m", "f", "p"))
    t = rng.choice(_HI_TEMPLATES)
    return t.format(
        s=rng.choice(_HI_SUBJ[g]),
        s2=rng.choice(_HI_SUBJ[g]),
        o=rng.choice(_HI_OBJ),
        adj=rng.choice(_HI_ADJ),
        adv=rng.choice(_HI_ADV),
        v=rng.choice(_HI_VERB[g]),
    )


def _kannada_sentence(rng: random.Random) -> str:
    g = rng.choice(("1", "2", "m", "f", "p"))
    t = rng.choice(_KN_TEMPLATES)
    return t.format(
        s=rng.choice(_KN_SUBJ[g]),
        s2=rng.choice(_KN_SUBJ[g]),
        o=rng.choice(_KN_OBJ),
        adj=rng.choice(_KN_ADJ),
        adv=rng.choice(_KN_ADV),
        v=rng.choice(_KN_VERB[g]),
    )


def _hinglish_sentence(rng: random.Random) -> str:
    t = rng.choice(_HINGLISH_TEMPLATES)
    return t.format(
        pron=rng.choice(_HI_ROM_PRON),
        obj=rng.choice(_HI_ROM_OBJ),
        adj=rng.choice(_HI_ROM_ADJ),
        vinf=rng.choice(_HI_ROM_VINF),
        vpast=rng.choice(_HI_ROM_VPAST),
    )


def _kanglish_sentence(rng: random.Random) -> str:
    t = rng.choice(_KANGLISH_TEMPLATES)
    return t.format(
        pron=rng.choice(_KN_ROM_PRON),
        obj=rng.choice(_KN_ROM_OBJ),
        adj=rng.choice(_KN_ROM_ADJ),
        vinf=rng.choice(_KN_ROM_VINF),
        vpast=rng.choice(_KN_ROM_VPAST),
    )


_SENTENCE_BUILDERS = {
    "prose_en": _english_sentence,
    "prose_hi": _hindi_sentence,
    "prose_kn": _kannada_sentence,
    "hinglish": _hinglish_sentence,
    "kanglish": _kanglish_sentence,
}


def sentence_space_size() -> Dict[str, int]:
    """Combinatorial size per sentence kind: how many distinct sentences exist.

    A useful sanity bound. If this is comparable to the number of sentences
    generated, the corpus is repetitive and fertility measurements will be
    memorisation scores.
    """
    en = len(_EN["det"]) ** 2 * len(_EN["adj"]) ** 2 * len(_EN["noun"]) ** 2 * len(_EN["verb"])
    hi = sum(
        len(_HI_SUBJ[g]) * len(_HI_VERB[g]) * len(_HI_OBJ) * len(_HI_ADJ) * len(_HI_ADV)
        for g in ("m", "f", "p")
    )
    kn = sum(
        len(_KN_SUBJ[g]) * len(_KN_VERB[g]) * len(_KN_OBJ) * len(_KN_ADJ) * len(_KN_ADV)
        for g in ("1", "2", "m", "f", "p")
    )
    hing = (
        len(_HI_ROM_PRON) * len(_HI_ROM_OBJ) * len(_HI_ROM_ADJ)
        * len(_HI_ROM_VINF) * len(_HI_ROM_VPAST)
    )
    kang = (
        len(_KN_ROM_PRON) * len(_KN_ROM_OBJ) * len(_KN_ROM_ADJ)
        * len(_KN_ROM_VINF) * len(_KN_ROM_VPAST)
    )
    return {
        "prose_en": en,
        "prose_hi": hi,
        "prose_kn": kn,
        "hinglish": hing,
        "kanglish": kang,
    }


# --------------------------------------------------------------------------
# Corpus generation
# --------------------------------------------------------------------------


def generate_corpus(target_words: int = 10000, seed: int = 1337) -> List[Document]:
    """Generate a code-switched multilingual corpus. Deterministic in (target, seed).

    Prose sentences are composed combinatorially, so different seeds produce
    largely disjoint text at the sentence level.
    """
    rng = random.Random(seed)
    docs: List[Document] = []
    words = 0
    idx = 0

    def add(kind: str, text: str) -> None:
        nonlocal words, idx
        docs.append(Document(f"syn-{idx:06d}", kind, text))
        idx += 1
        words += len(text.split())

    while words < target_words:
        pick = rng.random()

        if pick < 0.17:
            n = rng.randint(2, 5)
            add("prose_hi", " ".join(_hindi_sentence(rng) for _ in range(n)))
        elif pick < 0.32:
            n = rng.randint(2, 5)
            add("prose_kn", " ".join(_kannada_sentence(rng) for _ in range(n)))
        elif pick < 0.52:
            n = rng.randint(2, 5)
            add("prose_en", " ".join(_english_sentence(rng) for _ in range(n)))
        elif pick < 0.62:
            n = rng.randint(2, 4)
            add("hinglish", " ".join(_hinglish_sentence(rng) for _ in range(n)))
        elif pick < 0.70:
            n = rng.randint(2, 4)
            add("kanglish", " ".join(_kanglish_sentence(rng) for _ in range(n)))
        elif pick < 0.77:
            add("code", rng.choice(_CODE_BLOCKS))
        elif pick < 0.84:
            body = " ".join(rng.sample(_MD_LINES, k=rng.randint(4, 8)))
            if rng.random() < 0.6:
                body += "\n\n" + rng.choice(_MATH_BLOCKS)
            add("markdown", body)
        elif pick < 0.91:
            add("math", rng.choice(_MATH_BLOCKS) + " and " + rng.choice(_INLINE_MATH))
        else:
            tmpl = rng.choice(_MIXED_TEMPLATES)
            text = tmpl.format(
                hi=_hindi_sentence(rng),
                kn=_kannada_sentence(rng),
                en=_english_sentence(rng),
                emoji=rng.choice(_EMOJI),
            )
            add("mixed", text)

    return docs


def generate_heldout_corpus(
    target_words: int = 3000, seed: int = 20250
) -> List[Document]:
    """A disjoint, still-natural evaluation set.

    Different seed, same grammar, combinatorial sentence space -- so sentences are
    novel at the sequence level without being unnatural. This is strictly better
    than shuffling words, which destroys the multi-word regularities that
    whitespace ownership exists to capture and therefore biases the comparison.
    """
    return generate_corpus(target_words=target_words, seed=seed)


def sentence_overlap(a: Sequence[Document], b: Sequence[Document]) -> float:
    """Fraction of `b`'s sentences that also appear in `a`. Held-out sanity check."""
    sa = {s for d in a for s in d.text.split("।")}
    sb = [s for d in b for s in d.text.split("।")]
    if not sb:
        return 0.0
    hits = sum(1 for s in sb if s in sa)
    return hits / len(sb)


def write_documents(docs: Sequence[Document], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.as_dict(), ensure_ascii=False) + "\n")


def read_documents(path: str) -> List[Document]:
    out: List[Document] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(Document(**json.loads(line)))
    return out


def extract_slot_corpora(
    docs: Iterable[Document],
    out_dir: str,
    slot_map: Optional[Mapping[str, str]] = None,
    structural_slot: str = CODE_MATH,
    whitespace_ownership: str = DEFAULT_WHITESPACE_OWNERSHIP,
) -> Dict[str, int]:
    """Route every document into per-slot corpora, one JSONL file per slot.

    This is step 3a of the original design: scan the text, split it by class, and
    land each class in its own file. JSONL rather than plain text so that runs
    containing newlines (fenced code blocks) survive the round trip to disk.

    Returns {slot_name: byte_count} which drives budget allocation.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Clear stale slot corpora. Changing the slot config (e.g. grouping
    # DEVANAGARI+KANNADA into INDIC_SHARED) leaves files for slots that no longer
    # exist; reusing them silently would train on the wrong data or mask an
    # empty-slot misconfiguration.
    for stale in globlib.glob(os.path.join(out_dir, "*.jsonl")):
        os.remove(stale)

    handles: Dict[str, "object"] = {}
    counts: Dict[str, int] = {}
    try:
        for doc in docs:
            for run in segment(
                doc.text,
                slot_map=slot_map,
                structural_slot=structural_slot,
                whitespace_ownership=whitespace_ownership,
            ):
                fh = handles.get(run.slot)
                if fh is None:
                    fh = open(
                        os.path.join(out_dir, f"{run.slot}.jsonl"),
                        "w",
                        encoding="utf-8",
                    )
                    handles[run.slot] = fh
                    counts[run.slot] = 0
                fh.write(json.dumps(run.text, ensure_ascii=False) + "\n")
                counts[run.slot] += len(run.text.encode("utf-8"))
    finally:
        for fh in handles.values():
            fh.close()
    return counts


def read_slot_corpus(path: str) -> List[str]:
    """Read a slot corpus file into a list of runs."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_heldout_docs(docs: Sequence[Document], seed: int = 9999) -> List[Document]:
    """Fallback held-out builder for corpora we did not generate.

    Shuffles word order within each document. Words stay in vocabulary, so subword
    quality is measured normally, but sequences are new so whole-sentence
    memorisation gets no credit.

    Prefer `generate_heldout_corpus` for synthetic builds: word shuffling destroys
    multi-word structure and therefore biases the whitespace-ownership comparison.
    Use this only for real corpora, where a natural held-out split is unavailable.
    """
    rng = random.Random(seed)
    out: List[Document] = []
    for d in docs:
        if d.kind in ("code", "math", "markdown"):
            continue
        ws = d.text.split()
        if len(ws) < 3:
            continue
        rng.shuffle(ws)
        out.append(Document(f"held-{d.doc_id}", d.kind + "_held", " ".join(ws)))
    return out


# --------------------------------------------------------------------------
# Real-corpus registry (SPEC.md section 9)
# --------------------------------------------------------------------------


@dataclass
class CorpusEntry:
    name: str
    slot: str
    paths: Sequence[str]
    glob: str = "**/*.txt"
    encoding: str = "utf-8"
    weight: float = 1.0
    generator: Optional[str] = None
    size: int = 0
    kind_hint: Optional[str] = None  # e.g. "code" -> routes whole file to CODE_MATH


def load_registry(path: str) -> List[CorpusEntry]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    out: List[CorpusEntry] = []
    for item in raw.get("corpora", []):
        out.append(
            CorpusEntry(
                name=item["name"],
                slot=item.get("slot", ""),
                paths=list(item.get("paths", [])),
                glob=item.get("glob", "**/*.txt"),
                encoding=item.get("encoding", "utf-8"),
                weight=float(item.get("weight", 1.0)),
                generator=item.get("generator"),
                size=int(item.get("size", 0)),
                kind_hint=item.get("kind_hint"),
            )
        )
    return out


def iter_corpus_files(entry: CorpusEntry) -> Iterable[str]:
    for base in entry.paths:
        pattern = os.path.join(base, entry.glob)
        for p in sorted(globlib.iglob(pattern, recursive=True)):
            if os.path.isfile(p):
                yield p

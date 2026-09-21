"""Does seeding `initial_alphabet` actually guarantee those bytes survive training?

If the trainer prunes seeded tokens that never occur in the slot's corpus, the
`observed+ascii` margin is best-effort rather than a guarantee, and the bisection
fallback is what actually keeps losslessness and token cost bounded.
"""
import os
import tempfile

from tokenizer.build import build
from tokenizer.partitioned import PartitionedTokenizer
from tokenizer.train import BYTE_ALPHABET, UnrepresentableRun

WS = (0x09, 0x0A, 0x0D, 0x20)
rows = []
for i in range(5):
    d = tempfile.mkdtemp()
    out = os.path.join(d, "art")
    build(
        out_dir=out,
        data_dir=os.path.join(d, "data"),
        target_words=1500,
        total_vocab=3000,
        min_slot=300,
        max_slot=1200,
        verbose=False,
    )
    tok = PartitionedTokenizer.from_dir(out)
    missing = {}
    for name in ("DEVANAGARI", "KANNADA", "LATIN", "NEUTRAL"):
        v = set(tok.slots[name].local_vocab())
        gone = [hex(b) for b in WS if BYTE_ALPHABET[b] not in v]
        if gone:
            missing[name] = gone
    try:
        tok.slots["DEVANAGARI"].encode("\nनमस्ते")
        nl = "ok"
    except UnrepresentableRun:
        nl = "FALLBACK"
    rows.append((i, nl, missing))
    print("build %d: newline-encode=%-8s missing-ws-bytes=%s" % (i, nl, missing))

print()
print("builds where newline needed fallback:",
      sum(1 for _, nl, _ in rows if nl == "FALLBACK"), "/", len(rows))
print("builds with any missing seeded whitespace byte:",
      sum(1 for _, _, m in rows if m), "/", len(rows))

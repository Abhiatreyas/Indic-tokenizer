"""Freeze the merged vocabulary and its metadata.

The freeze step is the boundary between "we are still deciding" and "this is the
contract". After it, ids are stable and a checkpoint can depend on them
(SPEC.md section 4).

Artifacts written here are everything the decoder needs and nothing else: the
vocab, the id layout, the slot specs (which carry the script -> slot routing),
the per-slot models, and a manifest that hashes all of it.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Sequence

from .slots import (
    DEFAULT_EMBED_MULTIPLE,
    DEFAULT_SPECS,
    IdLayout,
    SlotSpec,
    validate_specs,
)
from .scripts import WHITESPACE_NEUTRAL

TOKENIZER_VERSION = "v1"

# Global control tokens. These live only in the merged layout, at ids 0..k-1,
# and are never produced by encoding content. Add SLOT_SEP here if you want
# opt-in in-stream slot markers; it costs exactly one row.
DEFAULT_SPECIALS: Sequence[str] = ("<pad>", "<bos>", "<eos>", "<unk>", "<mask>")

MANIFEST_NAME = "tokenizer_v1.manifest.json"
VOCAB_NAME = "vocab.v1.json"
SLOTS_NAME = "slots.v1.json"
SPECS_NAME = "slot_specs.v1.json"
MODELS_DIRNAME = "slot_models"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(
    out_dir: str,
    specials: Sequence[str],
    slots: Mapping[str, Any],
    specs: Sequence[SlotSpec] = DEFAULT_SPECS,
    extra: Mapping | None = None,
    embed_multiple: int = DEFAULT_EMBED_MULTIPLE,
    emit_slot_markers: bool = False,
    whitespace_ownership: str = WHITESPACE_NEUTRAL,
) -> Dict:
    """Write the vocab, layout, specs, per-slot models, and manifest.

    `slots` maps slot name -> object with .local_size / .local_vocab() / .save().
    """
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, MODELS_DIRNAME)
    os.makedirs(models_dir, exist_ok=True)

    validate_specs(specs)

    sizes = {spec.name: slots[spec.name].local_size for spec in specs}
    layout = IdLayout.from_sizes(
        specials, sizes, specs, embed_multiple=embed_multiple
    )

    # id -> token string. A LIST, not a dict: the same string may legitimately
    # appear in more than one slot (SPEC.md 6.2), and a dict would silently
    # collapse those duplicates.
    tokens: List[str] = [""] * layout.vocab_size
    for i, s in enumerate(specials):
        tokens[i] = s

    for spec in specs:
        rng = layout.slot_by_name(spec.name)
        vocab = slots[spec.name].local_vocab()
        if len(vocab) != rng.size:
            raise ValueError(
                f"slot {spec.name!r}: local vocab has {len(vocab)} entries but "
                f"the layout allocated {rng.size}"
            )
        for local_id, tok in enumerate(vocab):
            tokens[rng.start + local_id] = tok

    if any(t == "" for t in tokens):
        missing = [i for i, t in enumerate(tokens) if t == ""][:10]
        raise ValueError(f"unfilled token ids after merge: {missing}")

    vocab_path = os.path.join(out_dir, VOCAB_NAME)
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tokenizer_version": TOKENIZER_VERSION,
                "special_tokens": list(specials),
                "vocab_size": layout.vocab_size,
                "first_invalid_id": layout.first_invalid_id,
                "padded_vocab_size": layout.padded_vocab_size,
                "embed_multiple": layout.embed_multiple,
                "emit_slot_markers": emit_slot_markers,
                "whitespace_ownership": whitespace_ownership,
                "tokens": tokens,
            },
            fh,
            ensure_ascii=False,
        )
        fh.write("\n")

    slots_path = os.path.join(out_dir, SLOTS_NAME)
    layout.to_json(slots_path)

    # Script routing lives in the specs; the decoder needs it to segment.
    specs_path = os.path.join(out_dir, SPECS_NAME)
    with open(specs_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"slots": [s.as_dict() for s in specs]},
            fh,
            indent=2,
            ensure_ascii=False,
        )
        fh.write("\n")

    artifacts: Dict[str, str] = {}
    for spec in specs:
        if spec.algorithm == "bytes":
            continue
        path = os.path.join(models_dir, f"{spec.name}.json")
        slots[spec.name].save(path)
        artifacts[spec.name] = os.path.relpath(path, out_dir).replace("\\", "/")

    manifest = {
        "tokenizer_version": TOKENIZER_VERSION,
        "vocab_size": layout.vocab_size,
        "first_invalid_id": layout.first_invalid_id,
        "padded_vocab_size": layout.padded_vocab_size,
        "embed_multiple": layout.embed_multiple,
        "special_tokens": list(specials),
        "slot_sizes": sizes,
        "slot_algorithms": {s.name: s.algorithm for s in specs},
        "slot_scripts": {s.name: list(s.scripts) for s in specs},
        "slot_artifacts": artifacts,
        "files": {
            VOCAB_NAME: sha256_file(vocab_path),
            SLOTS_NAME: sha256_file(slots_path),
            SPECS_NAME: sha256_file(specs_path),
        },
    }
    if extra:
        manifest.update(extra)

    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return manifest


def verify_manifest(out_dir: str) -> Dict:
    """Re-hash the artifacts and compare against the manifest (gate G6)."""
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    problems: List[str] = []
    for name, expected in manifest.get("files", {}).items():
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            problems.append(f"{name}: missing")
            continue
        if sha256_file(path) != expected:
            problems.append(f"{name}: hash mismatch")

    for slot, rel in manifest.get("slot_artifacts", {}).items():
        if not os.path.exists(os.path.join(out_dir, rel)):
            problems.append(f"{rel}: missing")

    manifest["problems"] = problems
    manifest["ok"] = not problems
    return manifest


def load_specs(out_dir: str) -> tuple[SlotSpec, ...]:
    with open(os.path.join(out_dir, SPECS_NAME), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    specs = tuple(
        SlotSpec(
            name=it["name"],
            algorithm=it["algorithm"],
            scripts=tuple(it.get("scripts", ())),
            structural=bool(it.get("structural", False)),
            trained=bool(it.get("trained", True)),
            exempt_from_budget=bool(it.get("exempt_from_budget", False)),
            max_piece_length=int(it.get("max_piece_length", 48)),
        )
        for it in raw["slots"]
    )
    validate_specs(specs)
    return specs

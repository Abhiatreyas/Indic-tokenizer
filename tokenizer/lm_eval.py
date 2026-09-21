"""Bits-per-byte evaluation -- the decider.

Fertility (tokens/char) is a proxy. Bits-per-byte is the actual question: given
the same compute, which tokenizer lets a model predict the same text better? It is
comparable across tokenizers with different vocabularies precisely because it
normalises by the *bytes of the original text*, not by tokens.

Protocol (iso-compute):

  * identical model architecture and config for both tokenizers;
  * identical number of training tokens processed (steps x batch x seq_len);
  * identical training text, identical held-out text;
  * bpb = (sum of negative log-likelihood in nats / ln 2) / total bytes.

Why fix *tokens* rather than bytes: at pretraining time compute is the scarce
resource, and sequence length is the compute. Fixing the token budget means the
tokenizer with lower fertility reads more text for the same compute -- which is
exactly the effect under test. Bytes covered and parameters are reported so the
trade is visible rather than hidden.

Because these models are small, single-run bpb is noisy. Every comparison is run
over several seeds and reported as mean +/- spread; a difference smaller than the
spread is not a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _torch():
    import torch  # imported lazily so the rest of the package works without it

    return torch


@dataclass
class LMConfig:
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 512
    seq_len: int = 128
    batch_size: int = 16
    steps: int = 400
    lr: float = 3e-3
    warmup: int = 40
    dropout: float = 0.0
    weight_decay: float = 0.01

    def as_dict(self) -> dict:
        return asdict(self)


class TinyLM:
    """Factory for a small decoder-only transformer with weight-tied embeddings."""

    @staticmethod
    def build(vocab_size: int, cfg: LMConfig, seed: int):
        torch = _torch()
        torch.manual_seed(seed)

        class Block(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln1 = torch.nn.LayerNorm(cfg.d_model)
                self.attn = torch.nn.MultiheadAttention(
                    cfg.d_model, cfg.n_head, dropout=cfg.dropout, batch_first=True
                )
                self.ln2 = torch.nn.LayerNorm(cfg.d_model)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(cfg.d_model, cfg.d_ff),
                    torch.nn.GELU(),
                    torch.nn.Linear(cfg.d_ff, cfg.d_model),
                )

            def forward(self, x, mask):
                h = self.ln1(x)
                a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
                x = x + a
                return x + self.mlp(self.ln2(x))

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tok = torch.nn.Embedding(vocab_size, cfg.d_model)
                self.pos = torch.nn.Embedding(cfg.seq_len, cfg.d_model)
                self.blocks = torch.nn.ModuleList([Block() for _ in range(cfg.n_layer)])
                self.ln_f = torch.nn.LayerNorm(cfg.d_model)
                self.head = torch.nn.Linear(cfg.d_model, vocab_size, bias=False)
                # Weight tying: the embedding table is the only place the
                # tokenizer's vocabulary size enters the parameter count.
                self.head.weight = self.tok.weight
                self.register_buffer(
                    "mask",
                    torch.triu(
                        torch.full((cfg.seq_len, cfg.seq_len), float("-inf")),
                        diagonal=1,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "positions", torch.arange(cfg.seq_len), persistent=False
                )

            def forward(self, idx):
                b, t = idx.shape
                x = self.tok(idx) + self.pos(self.positions[:t])
                for blk in self.blocks:
                    x = blk(x, self.mask[:t, :t])
                return self.head(self.ln_f(x))

        return Model()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _sample_batch(tokens, cfg: LMConfig, rng: np.random.Generator, device):
    torch = _torch()
    n = len(tokens)
    hi = n - cfg.seq_len - 1
    if hi <= 0:
        raise ValueError(
            f"token stream too short ({n}) for seq_len={cfg.seq_len}"
        )
    starts = rng.integers(0, hi, size=cfg.batch_size)
    x = np.stack([tokens[s : s + cfg.seq_len] for s in starts])
    y = np.stack([tokens[s + 1 : s + cfg.seq_len + 1] for s in starts])
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )


def train_lm(
    train_tokens: Sequence[int],
    vocab_size: int,
    cfg: LMConfig,
    seed: int,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[object, Dict]:
    """Train a tiny LM on a token stream. Returns (model, info)."""
    torch = _torch()
    tokens = np.asarray(train_tokens, dtype=np.int64)
    rng = np.random.default_rng(seed)
    model = TinyLM.build(vocab_size, cfg, seed).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup))
    )
    lossf = torch.nn.CrossEntropyLoss()
    model.train()

    first, last = float("nan"), float("nan")
    for step in range(cfg.steps):
        x, y = _sample_batch(tokens, cfg, rng, device)
        logits = model(x)
        loss = lossf(logits.reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
        if verbose and (step + 1) % 100 == 0:
            print(f"      step {step + 1}/{cfg.steps} loss={last:.4f}")

    tokens_seen = cfg.steps * cfg.batch_size * cfg.seq_len
    info = {
        "seed": seed,
        "steps": cfg.steps,
        "tokens_seen": tokens_seen,
        "first_loss_nats": first,
        "final_loss_nats": last,
        "params": count_params(model),
        "vocab_size": vocab_size,
        "stream_tokens": int(tokens.shape[0]),
    }
    return model, info


def bits_per_byte(
    model, tokens: Sequence[int], n_bytes: int, cfg: LMConfig, device: str = "cpu"
) -> Dict:
    """Score a held-out token stream as bits per byte of the original text.

    Contiguous non-overlapping windows. Window k's first token is scored as window
    k-1's final position, so exactly one token of the entire stream (its first)
    goes unscored -- identical treatment for every tokenizer. The denominator is
    the byte length of the whole held-out text, so nothing is dropped from it.
    """
    torch = _torch()
    arr = np.asarray(tokens, dtype=np.int64)
    if len(arr) < 2:
        raise ValueError("held-out stream too short to score")
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")

    model.eval()
    total_nll = 0.0
    scored = 0
    with torch.no_grad():
        for start in range(0, len(arr) - 1, cfg.seq_len):
            chunk = arr[start : start + cfg.seq_len + 1]
            if len(chunk) < 2:
                break
            x = torch.from_numpy(chunk[:-1]).unsqueeze(0).to(device)
            y = torch.from_numpy(chunk[1:]).unsqueeze(0).to(device)
            logits = model(x)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"
            )
            total_nll += float(nll.item())
            scored += int(y.numel())

    bpb = (total_nll / math.log(2)) / n_bytes
    return {
        "bits_per_byte": bpb,
        "total_nll_nats": total_nll,
        "scored_tokens": scored,
        "bytes": n_bytes,
        "tokens_per_byte": scored / n_bytes,
        "perplexity_per_token": math.exp(total_nll / max(1, scored)),
    }


def compare_bpb(
    partitioned_tokens: Sequence[int],
    baseline_tokens: Sequence[int],
    partitioned_vocab: int,
    baseline_vocab: int,
    heldout_partitioned: Sequence[int],
    heldout_baseline: Sequence[int],
    heldout_bytes: int,
    cfg: LMConfig,
    seeds: Sequence[int] = (1, 2, 3),
    device: str = "cpu",
    verbose: bool = False,
) -> Dict:
    """Train both models on the same token budget and compare held-out bpb."""
    runs: List[Dict] = []
    for seed in seeds:
        pm, pinfo = train_lm(
            partitioned_tokens, partitioned_vocab, cfg, seed, device, verbose
        )
        pb = bits_per_byte(pm, heldout_partitioned, heldout_bytes, cfg, device)

        bm, binfo = train_lm(
            baseline_tokens, baseline_vocab, cfg, seed, device, verbose
        )
        bb = bits_per_byte(bm, heldout_baseline, heldout_bytes, cfg, device)

        runs.append(
            {
                "seed": seed,
                "partitioned": {**pinfo, **pb},
                "baseline": {**binfo, **bb},
                "bpb_delta": bb["bits_per_byte"] - pb["bits_per_byte"],
            }
        )
        if verbose:
            print(
                f"    seed {seed}: partitioned={pb['bits_per_byte']:.4f} "
                f"baseline={bb['bits_per_byte']:.4f} "
                f"delta={runs[-1]['bpb_delta']:+.4f}"
            )

    def stats(key: str, sub: str):
        vals = [r[key][sub] for r in runs]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }

    p_bpb = stats("partitioned", "bits_per_byte")
    b_bpb = stats("baseline", "bits_per_byte")
    deltas = [r["bpb_delta"] for r in runs]
    delta_std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

    # A win counts only if the mean delta exceeds the run-to-run spread.
    decisive = abs(float(np.mean(deltas))) > delta_std

    return {
        "config": cfg.as_dict(),
        "seeds": list(seeds),
        "runs": runs,
        "partitioned_bpb": p_bpb,
        "baseline_bpb": b_bpb,
        "bpb_delta_mean": float(np.mean(deltas)),
        "bpb_delta_std": delta_std,
        "decisive": decisive,
        "winner": (
            "partitioned"
            if float(np.mean(deltas)) > 0 and decisive
            else "baseline"
            if float(np.mean(deltas)) < 0 and decisive
            else "inconclusive"
        ),
        "partitioned_params": runs[0]["partitioned"]["params"],
        "baseline_params": runs[0]["baseline"]["params"],
        "partitioned_stream_tokens": runs[0]["partitioned"]["stream_tokens"],
        "baseline_stream_tokens": runs[0]["baseline"]["stream_tokens"],
    }

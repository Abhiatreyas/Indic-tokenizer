"""Bits-per-byte correctness.

bpb is the decision metric, so the math is tested against closed-form values
rather than eyeballed. A wrong denominator or a log-base slip would silently
invert the verdict.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tokenizer.lm_eval import (  # noqa: E402
    LMConfig,
    bits_per_byte,
    compare_bpb,
    count_params,
    train_lm,
)


class UniformModel(torch.nn.Module):
    """Emits identical logits for every position: NLL per token is exactly ln(V)."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        return torch.zeros(idx.shape[0], idx.shape[1], self.vocab_size)


class OracleModel(torch.nn.Module):
    """Predicts token+1 for a cyclic stream: bpb must be ~0.

    Position-independent on purpose, so it needs no knowledge of where a window
    started -- `bits_per_byte` does not expose that.
    """

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, idx):
        b, t = idx.shape
        logits = torch.full((b, t, self.vocab), -50.0)
        nxt = (idx + 1) % self.vocab
        return logits.scatter(2, nxt.unsqueeze(-1), 50.0)


CFG = LMConfig(seq_len=32, batch_size=2, steps=2)


def test_uniform_model_matches_closed_form():
    """bpb = scored_tokens * ln(V) / ln(2) / bytes."""
    vocab = 8
    stream = list(range(vocab)) * 10
    n_bytes = 100
    out = bits_per_byte(UniformModel(vocab), stream, n_bytes, CFG)
    expected = out["scored_tokens"] * math.log(vocab) / math.log(2) / n_bytes
    # float32 logits, so exact equality is not available.
    assert out["bits_per_byte"] == pytest.approx(expected, rel=1e-5)
    assert out["perplexity_per_token"] == pytest.approx(vocab, rel=1e-4)


def test_denominator_is_bytes_not_tokens():
    """Same tokens, more bytes -> strictly lower bpb."""
    stream = list(range(6)) * 8
    a = bits_per_byte(UniformModel(6), stream, 100, CFG)
    b = bits_per_byte(UniformModel(6), stream, 200, CFG)
    assert b["bits_per_byte"] == pytest.approx(a["bits_per_byte"] / 2, rel=1e-9)


def test_oracle_model_gives_zero_bpb():
    vocab = 4
    stream = [i % vocab for i in range(200)]
    out = bits_per_byte(OracleModel(vocab), stream, 64, CFG)
    assert out["bits_per_byte"] == pytest.approx(0.0, abs=1e-6)


def test_only_the_first_token_of_the_stream_is_unscored():
    """Contiguous non-overlapping windows: window k's first token was already
    scored as window k-1's last position. So exactly one token is skipped, and
    the two sides of a comparison are treated identically."""
    cfg = LMConfig(seq_len=16, batch_size=2, steps=1)
    stream = list(range(5)) * 40  # 200 tokens
    out = bits_per_byte(UniformModel(5), stream, 400, cfg)
    assert out["scored_tokens"] == len(stream) - 1


def test_short_stream_rejected():
    with pytest.raises(ValueError, match="too short"):
        bits_per_byte(UniformModel(4), [1], 10, CFG)


def test_zero_bytes_rejected():
    with pytest.raises(ValueError, match="n_bytes"):
        bits_per_byte(UniformModel(4), [1, 2, 3], 0, CFG)


def test_training_reduces_loss_on_a_learnable_stream():
    """Guards the optimizer, masking and windowing plumbing."""
    cfg = LMConfig(
        d_model=32, n_layer=1, n_head=2, d_ff=64,
        seq_len=16, batch_size=4, steps=60, lr=5e-3, warmup=5,
    )
    # Highly predictable: a repeating cycle.
    stream = [i % 4 for i in range(2000)]
    _, info = train_lm(stream, 4, cfg, seed=0)
    assert info["final_loss_nats"] < info["first_loss_nats"]


def test_stream_too_short_for_seq_len_rejected():
    cfg = LMConfig(seq_len=64, batch_size=2, steps=1)
    with pytest.raises(ValueError, match="token stream too short"):
        train_lm([1, 2, 3], 4, cfg, seed=0)


def test_param_count_scales_with_vocab():
    a = count_params(_tiny_model(vocab_size=64))
    b = count_params(_tiny_model(vocab_size=256))
    assert b > a


def _tiny_model(vocab_size: int):
    from tokenizer.lm_eval import TinyLM

    return TinyLM.build(vocab_size, LMConfig(d_model=32, n_layer=1, n_head=2, d_ff=64), 0)


def test_count_params_matches_hand_computation():
    """Weight tying means the vocab enters the count exactly once."""
    vocab, d = 100, 32
    cfg = LMConfig(d_model=d, n_layer=0, n_head=2, d_ff=64, seq_len=8)
    model = __import__("tokenizer.lm_eval", fromlist=["TinyLM"]).TinyLM.build(vocab, cfg, 0)
    # tok(vocab*d) + pos(seq*d) + ln_f(2d) ; head is tied to tok
    expected = vocab * d + cfg.seq_len * d + 2 * d
    assert count_params(model) == expected


def test_compare_bpb_reports_winners_and_spread():
    p_train = [i % 10 for i in range(1500)]
    b_train = [i % 12 for i in range(1500)]
    p_held = [i % 10 for i in range(300)]
    b_held = [i % 12 for i in range(300)]
    cfg = LMConfig(
        d_model=32, n_layer=1, n_head=2, d_ff=64,
        seq_len=32, batch_size=4, steps=25, warmup=5,
    )
    out = compare_bpb(
        p_train, b_train, 10, 12, p_held, b_held, 600, cfg, seeds=(1, 2)
    )
    assert set(out["baseline_bpb"]) == {"mean", "std", "values"}
    assert len(out["runs"]) == 2
    assert out["winner"] in ("partitioned", "baseline", "inconclusive")
    assert isinstance(out["decisive"], bool)
    assert out["partitioned_params"] > 0 and out["baseline_params"] > 0


def test_identical_tokenizers_are_inconclusive():
    """A sanity floor: comparing a tokenizer with itself must not declare a win."""
    train = [i % 10 for i in range(1500)]
    held = [i % 10 for i in range(300)]
    cfg = LMConfig(
        d_model=32, n_layer=1, n_head=2, d_ff=64,
        seq_len=32, batch_size=4, steps=25, warmup=5,
    )
    out = compare_bpb(train, train, 10, 10, held, held, 600, cfg, seeds=(1, 2))
    # Same data, same seeds -> identical bpb, delta exactly zero.
    assert out["bpb_delta_mean"] == pytest.approx(0.0, abs=1e-9)
    assert out["winner"] == "inconclusive"

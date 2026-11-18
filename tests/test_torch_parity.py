"""Cross-validation against PyTorch autograd (optional, skipped without torch).

This is the strongest possible check of the hand-written backward: the same
weights, tokens and targets are pushed through torch's autograd and through
nimbus-train's manual backward, and every gradient must agree to float64
machine precision.
"""
import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train import model as M

torch = pytest.importorskip("torch")


def _gpt2() -> ModelConfig:
    return ModelConfig(name="tp-gpt2", vocab_size=64, n_layers=2, n_heads=4, d_model=32,
                       d_mlp=96, max_seq_len=16)


def _llama() -> ModelConfig:
    return ModelConfig(name="tp-llama", vocab_size=64, n_layers=2, n_heads=4, n_kv_heads=2,
                       d_model=32, d_mlp=88, max_seq_len=16, norm="rmsnorm",
                       activation="swiglu", pos_encoding="rope", tie_embeddings=False,
                       norm_eps=1e-6)


def _torch_model(cfg: ModelConfig, params: dict):
    import torch

    T = 10
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, cfg.vocab_size, size=(2, T))
    targets = rng.integers(0, cfg.vocab_size, size=(2, T))
    tok = torch.tensor(tokens, dtype=torch.long)
    tgt = torch.tensor(targets, dtype=torch.long)

    B, d, hd = 2, cfg.d_model, cfg.resolved_head_dim
    H, KV = cfg.n_heads, cfg.resolved_n_kv_heads
    eps = cfg.norm_eps
    scale = cfg.attn_scale

    def norm_fn(x, prefix):
        if cfg.norm == "rmsnorm":
            ms = (x * x).mean(-1, keepdim=True)
            return x / torch.sqrt(ms + eps) * tp[prefix + ".weight"]
        return (x - x.mean(-1, keepdim=True)) / torch.sqrt(
            x.var(-1, keepdim=True, correction=0) + eps
        ) * tp[prefix + ".weight"] + tp[prefix + ".bias"]

    x = tp["wte"][tok]
    if cfg.pos_encoding == "learned":
        x = x + tp["wpe"][:T]
    cos_t = sin_t = None
    if cfg.pos_encoding == "rope":
        cos_t, sin_t = M.O.build_rope(hd, cfg.max_seq_len, cfg.rope_theta)
        cos_t, sin_t = cos_t[:T][None, None, :, :], sin_t[:T][None, None, :, :]

    for i in range(cfg.n_layers):
        pre = f"h.{i}."
        h = norm_fn(x, pre + "norm1")
        q = (h @ tp[pre + "attn.q.weight"]).reshape(B, T, H, hd).transpose(1, 2)
        k = (h @ tp[pre + "attn.k.weight"]).reshape(B, T, KV, hd).transpose(1, 2)
        v = (h @ tp[pre + "attn.v.weight"]).reshape(B, T, KV, hd).transpose(1, 2)
        if cos_t is not None:
            def rope(z):
                half = z.shape[-1] // 2
                z1, z2 = z[..., :half], z[..., half:]
                return torch.cat([z1 * torch.tensor(cos_t) - z2 * torch.tensor(sin_t),
                                  z1 * torch.tensor(sin_t) + z2 * torch.tensor(cos_t)], dim=-1)
            q, k = rope(q), rope(k)
        if cfg.kv_group_size > 1:
            k = k.repeat_interleave(cfg.kv_group_size, dim=1)
            v = v.repeat_interleave(cfg.kv_group_size, dim=1)
        scores = (q @ k.transpose(-1, -2)) * scale
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        scores = torch.where(mask, scores, torch.tensor(-1e9, dtype=scores.dtype))
        probs = torch.softmax(scores, dim=-1)
        attn = (probs @ v).transpose(1, 2).reshape(B, T, H * hd)
        x_mid = x + attn @ tp[pre + "attn.o.weight"]
        h2 = norm_fn(x_mid, pre + "norm2")
        if cfg.activation == "swiglu":
            g = torch.nn.functional.silu(h2 @ tp[pre + "mlp.gate.weight"])
            u = h2 @ tp[pre + "mlp.up.weight"]
            x = x_mid + (g * u) @ tp[pre + "mlp.down.weight"]
        else:
            c = 0.7978845608028654
            fc = h2 @ tp[pre + "mlp.fc.weight"]
            act = 0.5 * fc * (1 + torch.tanh(c * (fc + 0.044715 * fc ** 3)))
            x = x_mid + act @ tp[pre + "mlp.proj.weight"]

    x_f = norm_fn(x, "norm_f")
    w_head = tp["lm_head.weight"] if "lm_head.weight" in tp else tp["wte"].T
    logits = x_f.reshape(B * T, d) @ w_head
    loss = torch.nn.functional.cross_entropy(logits, tgt.reshape(-1))
    loss.backward()
    return float(loss.item()), tp, tokens, targets


@pytest.mark.parametrize("cfg", [_gpt2(), _llama()], ids=["gpt2", "llama"])
def test_gradients_match_torch_autograd(cfg):
    params64 = M.init_params(cfg, seed=0, dtype=np.float64)
    # torch parity runs in float64 via the same dict (torch casts freely)
    loss_np, grads = M.loss_and_grad(cfg, params64, *_tokens_and_targets(cfg, params64))
    loss_t, tp, tokens, targets = _torch_model(cfg, params64)
    assert abs(loss_np - loss_t) < 1e-9, (loss_np, loss_t)
    for k, g in grads.items():
        tgrad = tp[k].grad.numpy()
        denom = max(float(np.abs(tgrad).max()), 1e-8)
        rel = float(np.abs(g - tgrad).max()) / denom
        assert rel < 1e-9, f"param {k}: rel diff {rel:.2e}"


def _tokens_and_targets(cfg, params):
    rng = np.random.default_rng(0)
    return (rng.integers(0, cfg.vocab_size, size=(2, 10)),
            rng.integers(0, cfg.vocab_size, size=(2, 10)))

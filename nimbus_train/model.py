"""Decoder-only transformer with a fully hand-written backward pass.

Parameter names, storage layout ``(in_features, out_features)`` and
``config.json`` are identical to nimbus-infer, so a model trained here exports
straight into the inference engine (see ``export.py``).

Both architecture families are supported through :class:`~nimbus_train.config.ModelConfig`:

* GPT-2  -- learned positional embeddings, LayerNorm, tanh-GELU, MHA.
* LLaMA  -- RoPE, RMSNorm, SwiGLU, grouped-query attention (GQA).

The forward pass records a per-layer context of intermediates;
:func:`backward` walks it in reverse and produces exact gradients for every
parameter. No autograd -- every backward formula is derived by hand and
verified against finite differences in the test suite.

LoRA adapters are supported natively: adapter weights live in the same
parameter dict as ``<name>.lora_A`` / ``<name>.lora_B`` (see ``lora.py``) and
the forward/backward automatically include the low-rank delta. Passing
``train_only`` to :func:`backward` restricts gradient computation to those
keys, which also skips the frozen weights' gradient matmuls entirely.
"""
from __future__ import annotations

import math

import numpy as np

from . import ops as O
from .config import ModelConfig
from .ops import DTYPE


# ---------------------------------------------------------------------------
# Initialisation (GPT-2 recipe: normal(0, 0.02), residual projections scaled)
# ---------------------------------------------------------------------------
def init_params(config: ModelConfig, seed: int = 0, dtype=DTYPE) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = config.d_model
    hd = config.resolved_head_dim
    n_kv = config.resolved_n_kv_heads
    mlp = config.d_mlp
    res_scale = 1.0 / math.sqrt(2 * config.n_layers)  # nanoGPT / GPT-2 residual init

    def normal(*shape: int, scale: float = 0.02) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(dtype)

    params: dict[str, np.ndarray] = {"wte": normal(config.vocab_size, d)}
    if config.pos_encoding == "learned":
        params["wpe"] = normal(config.max_seq_len, d)

    for i in range(config.n_layers):
        p = f"h.{i}."
        params[p + "norm1.weight"] = np.ones(d, dtype=dtype)
        params[p + "norm2.weight"] = np.ones(d, dtype=dtype)
        if config.norm == "layernorm":
            params[p + "norm1.bias"] = np.zeros(d, dtype=dtype)
            params[p + "norm2.bias"] = np.zeros(d, dtype=dtype)

        params[p + "attn.q.weight"] = normal(d, config.n_heads * hd)
        params[p + "attn.k.weight"] = normal(d, n_kv * hd)
        params[p + "attn.v.weight"] = normal(d, n_kv * hd)
        params[p + "attn.o.weight"] = normal(config.n_heads * hd, d, scale=res_scale)

        if config.activation == "swiglu":
            params[p + "mlp.gate.weight"] = normal(d, mlp)
            params[p + "mlp.up.weight"] = normal(d, mlp)
            params[p + "mlp.down.weight"] = normal(mlp, d, scale=res_scale)
        else:
            params[p + "mlp.fc.weight"] = normal(d, mlp)
            params[p + "mlp.proj.weight"] = normal(mlp, d, scale=res_scale)

    params["norm_f.weight"] = np.ones(d, dtype=dtype)
    if config.norm == "layernorm":
        params["norm_f.bias"] = np.zeros(d, dtype=dtype)
    if not config.tie_embeddings:
        params["lm_head.weight"] = normal(d, config.vocab_size)
    return params


def _lm_head_weight(params: dict[str, np.ndarray]) -> np.ndarray:
    if "lm_head.weight" in params:
        return params["lm_head.weight"]
    return params["wte"].T  # weight tying


_BIAS_CACHE: dict[tuple[int, type], np.ndarray] = {}


def _causal_bias(T: int, dtype) -> np.ndarray:
    """Additive causal mask (0 keep, -1e9 drop), cached per (T, dtype)."""
    key = (T, dtype)
    if key not in _BIAS_CACHE:
        bias = np.zeros((T, T), dtype=dtype)
        bias[np.triu_indices(T, k=1)] = -1e9
        _BIAS_CACHE[key] = bias
    return _BIAS_CACHE[key]


# ---------------------------------------------------------------------------
# LoRA-aware linear sites
#   y = x @ W + (x @ A) @ B     (A: (in, r), B: (r, out); B == 0 at init)
# ---------------------------------------------------------------------------
def _site_forward(params, name, x, bias=None):
    y = x @ params[name]
    if bias is not None:
        y = y + bias
    a = params.get(name + ".lora_A")
    if a is not None:
        y = y + (x @ a) @ params[name + ".lora_B"]
    return y


def _site_backward(params, grads, name, x, dy, train_only=None):
    """Gradients for one LoRA-aware linear site; returns dx.

    When ``train_only`` is given, the frozen base weight's gradient matmul is
    skipped entirely (that is the point of LoRA on CPU).
    """
    W = params[name]
    a = params.get(name + ".lora_A")
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    if a is None:
        dx = dy @ W.T
        if train_only is None or name in train_only:
            grads[name] = x2.T @ dy2
        return dx
    b = params[name + ".lora_B"]
    dx = dy @ W.T + (dy @ b.T) @ a.T
    if train_only is None or name in train_only:
        grads[name] = x2.T @ dy2
    grads[name + ".lora_A"] = x2.T @ (dy2 @ b.T)
    grads[name + ".lora_B"] = (x2 @ a).T @ dy2
    return dx


def _site_bias_grad(params, grads, name, dy, train_only=None):
    if name + ".bias" in params and (train_only is None or name + ".bias" in train_only):
        grads[name + ".bias"] = dy.reshape(-1, dy.shape[-1]).sum(axis=0)


# ---------------------------------------------------------------------------
# Forward -- records per-layer intermediates when need_grad=True
# ---------------------------------------------------------------------------
def forward(
    config: ModelConfig,
    params: dict[str, np.ndarray],
    tokens: np.ndarray,
    need_grad: bool = True,
):
    """Batched forward pass. ``tokens`` is (batch, seq) integer ids.

    Returns ``(logits, ctx)`` where logits is (batch, seq, vocab). ``ctx``
    carries the intermediates the backward pass consumes.
    """
    cfg = config
    p = params
    tokens = np.asarray(tokens, dtype=np.int64)
    B, T = tokens.shape
    if T > cfg.max_seq_len:
        raise ValueError(f"seq {T} exceeds max_seq_len {cfg.max_seq_len}")

    d = cfg.d_model
    hd = cfg.resolved_head_dim
    H = cfg.n_heads
    KV = cfg.resolved_n_kv_heads
    scale = cfg.attn_scale
    eps = cfg.norm_eps

    ctx: dict = {"tokens": tokens, "layers": [{} for _ in range(cfg.n_layers)]}

    cos = sin = None
    if cfg.pos_encoding == "rope":
        cos_t, sin_t = O.build_rope(hd, cfg.max_seq_len, cfg.rope_theta)
        cos = cos_t[:T][None, None, :, :]  # broadcast over (batch, heads)
        sin = sin_t[:T][None, None, :, :]
        ctx["rope_cos"], ctx["rope_sin"] = cos, sin

    x = p["wte"][tokens]
    if cfg.pos_encoding == "learned":
        x = x + p["wpe"][:T]

    for i in range(cfg.n_layers):
        pre = f"h.{i}."
        L = ctx["layers"][i]
        x_in = x

        # ---- pre-attention norm ----
        if cfg.norm == "rmsnorm":
            h, aux1 = O.rms_norm_fwd(x, p[pre + "norm1.weight"], eps)
        else:
            h, aux1 = O.layer_norm_fwd(x, p[pre + "norm1.weight"], p[pre + "norm1.bias"], eps)
        if need_grad:
            L["aux1"], L["h"] = aux1, h

        # ---- attention ----
        q = _site_forward(p, pre + "attn.q.weight", h).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = _site_forward(p, pre + "attn.k.weight", h).reshape(B, T, KV, hd).transpose(0, 2, 1, 3)
        v = _site_forward(p, pre + "attn.v.weight", h).reshape(B, T, KV, hd).transpose(0, 2, 1, 3)

        if cos is not None:
            q, _ = O.rope_fwd(q, cos, sin)
            k, _ = O.rope_fwd(k, cos, sin)

        if cfg.kv_group_size > 1:
            k = np.repeat(k, cfg.kv_group_size, axis=1)
            v = np.repeat(v, cfg.kv_group_size, axis=1)

        scores = q @ k.transpose(0, 1, 3, 2)  # (B, H, T, T)
        scores *= scale
        scores += _causal_bias(T, scores.dtype)  # in-place additive mask
        probs = O.softmax_fwd(scores)  # destructive on scores

        attn_out = probs @ v  # (B, H, T, hd)
        attn_concat = attn_out.transpose(0, 2, 1, 3).reshape(B, T, H * hd)
        attn_proj = _site_forward(p, pre + "attn.o.weight", attn_concat)
        x_mid = x_in + attn_proj

        if need_grad:
            L["q"], L["k"], L["v"], L["probs"], L["attn_concat"], L["x_mid"] = q, k, v, probs, attn_concat, x_mid

        # ---- pre-mlp norm ----
        if cfg.norm == "rmsnorm":
            h2, aux2 = O.rms_norm_fwd(x_mid, p[pre + "norm2.weight"], eps)
        else:
            h2, aux2 = O.layer_norm_fwd(x_mid, p[pre + "norm2.weight"], p[pre + "norm2.bias"], eps)
        if need_grad:
            L["aux2"], L["h2"] = aux2, h2

        # ---- mlp ----
        if cfg.activation == "swiglu":
            g_pre = _site_forward(p, pre + "mlp.gate.weight", h2)
            u_pre = _site_forward(p, pre + "mlp.up.weight", h2)
            g, s = O.silu_fwd(g_pre)
            mid = g * u_pre
            out = _site_forward(p, pre + "mlp.down.weight", mid)
            if need_grad:
                L["g_pre"], L["s"], L["u_pre"], L["mid"] = g_pre, s, u_pre, mid
        else:
            fc_pre = _site_forward(p, pre + "mlp.fc.weight", h2)
            act, t = O.gelu_fwd(fc_pre)
            out = _site_forward(p, pre + "mlp.proj.weight", act)
            if need_grad:
                L["fc_pre"], L["t"], L["act"] = fc_pre, t, act
        x = x_mid + out

    # ---- final norm + head ----
    if cfg.norm == "rmsnorm":
        x_f, aux_f = O.rms_norm_fwd(x, p["norm_f.weight"], eps)
    else:
        x_f, aux_f = O.layer_norm_fwd(x, p["norm_f.weight"], p["norm_f.bias"], eps)
    if need_grad:
        ctx["aux_f"], ctx["x_f"] = aux_f, x_f

    logits = x_f.reshape(B * T, d) @ _lm_head_weight(p)
    return logits.reshape(B, T, -1), ctx


# ---------------------------------------------------------------------------
# Backward -- exact gradients for every parameter, walked in reverse
# ---------------------------------------------------------------------------
def backward(
    config: ModelConfig,
    params: dict[str, np.ndarray],
    ctx: dict,
    dlogits: np.ndarray,
    train_only: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Exact gradients. ``train_only`` restricts output keys (and skips the
    gradient matmuls of frozen weights -- used by LoRA fine-tuning)."""
    cfg = config
    p = params
    B, T = ctx["tokens"].shape
    d = cfg.d_model
    H = cfg.n_heads
    KV = cfg.resolved_n_kv_heads
    scale = cfg.attn_scale

    grads: dict[str, np.ndarray] = {}

    dl = dlogits.reshape(B * T, -1)
    x_f = ctx["x_f"].reshape(B * T, d)

    # ---- final norm + head ----
    if "lm_head.weight" in p:
        if train_only is None or "lm_head.weight" in train_only:
            grads["lm_head.weight"] = x_f.T @ dl
        dx = (dl @ p["lm_head.weight"].T).reshape(B, T, d)
    else:
        if train_only is None or "wte" in train_only:
            d_wte = dl.T @ x_f  # logits = x_f @ wte.T  ->  dwte = dl.T @ x_f
        else:
            d_wte = None
        dx = (dl @ p["wte"]).reshape(B, T, d)

    if cfg.norm == "rmsnorm":
        x_hat, inv_std = ctx["aux_f"]
        dx, dw = O.rms_norm_bwd(x_hat, inv_std, p["norm_f.weight"], dx)
        if train_only is None or "norm_f.weight" in train_only:
            grads["norm_f.weight"] = dw
    else:
        x_hat, inv_std = ctx["aux_f"]
        dx, dw, db = O.layer_norm_bwd(x_hat, inv_std, p["norm_f.weight"], dx)
        if train_only is None or "norm_f.weight" in train_only:
            grads["norm_f.weight"] = dw
        if "norm_f.bias" in p and (train_only is None or "norm_f.bias" in train_only):
            grads["norm_f.bias"] = db

    cos = ctx.get("rope_cos")
    sin = ctx.get("rope_sin")

    for i in reversed(range(cfg.n_layers)):
        pre = f"h.{i}."
        L = ctx["layers"][i]

        # ---- mlp ---- (x = x_mid + mlp_out)
        d_mlp_out = dx
        if cfg.activation == "swiglu":
            g_pre, s, u_pre, mid = L["g_pre"], L["s"], L["u_pre"], L["mid"]
            d_mid = _site_backward(p, grads, pre + "mlp.down.weight", mid, d_mlp_out, train_only)
            g = g_pre * s  # silu output, recomputed from the stored pre-activation
            dg = d_mid * u_pre
            du = d_mid * g
            d_gpre = O.silu_bwd(g_pre, s, dg)
            d_h2 = _site_backward(p, grads, pre + "mlp.gate.weight", L["h2"], d_gpre, train_only) \
                + _site_backward(p, grads, pre + "mlp.up.weight", L["h2"], du, train_only)
        else:
            # proj takes the GELU activation as input; fc takes the norm output
            d_act = _site_backward(p, grads, pre + "mlp.proj.weight", L["act"], d_mlp_out, train_only)
            d_fc_pre = O.gelu_bwd(L["fc_pre"], L["t"], d_act)
            d_h2 = _site_backward(p, grads, pre + "mlp.fc.weight", L["h2"], d_fc_pre, train_only)

        # ---- pre-mlp norm ----
        if cfg.norm == "rmsnorm":
            d_xmid_n, dw2 = O.rms_norm_bwd(L["aux2"][0], L["aux2"][1], p[pre + "norm2.weight"], d_h2)
            if train_only is None or pre + "norm2.weight" in train_only:
                grads[pre + "norm2.weight"] = dw2
        else:
            d_xmid_n, dw2, db2 = O.layer_norm_bwd(
                L["aux2"][0], L["aux2"][1], p[pre + "norm2.weight"], d_h2
            )
            if train_only is None or pre + "norm2.weight" in train_only:
                grads[pre + "norm2.weight"] = dw2
            if pre + "norm2.bias" in p and (train_only is None or pre + "norm2.bias" in train_only):
                grads[pre + "norm2.bias"] = db2

        # residual: x_mid = x_in + attn_proj  and  x = x_mid + mlp_out
        d_xmid = dx + d_xmid_n

        # ---- attention ----
        d_attn_concat = _site_backward(p, grads, pre + "attn.o.weight", L["attn_concat"], d_xmid, train_only)
        _site_bias_grad(p, grads, pre + "attn.o", d_attn_concat, train_only)

        probs, q, k, v = L["probs"], L["q"], L["k"], L["v"]
        dout_attn = d_attn_concat.reshape(B, T, H, cfg.resolved_head_dim).transpose(0, 2, 1, 3)
        # dv is read from probs, then softmax_bwd overwrites probs in place
        dv = probs.transpose(0, 1, 3, 2) @ dout_attn
        dprobs = dout_attn @ v.transpose(0, 1, 3, 2)
        dqk = O.softmax_bwd(probs, dprobs)  # destructive: dqk IS probs
        dqk *= scale
        dq = dqk @ k
        dk = dqk.transpose(0, 1, 3, 2) @ q

        if cfg.kv_group_size > 1:  # undo the GQA repeat
            g = cfg.kv_group_size
            dk = dk.reshape(B, KV, g, T, cfg.resolved_head_dim).sum(axis=2)
            dv = dv.reshape(B, KV, g, T, cfg.resolved_head_dim).sum(axis=2)

        if cos is not None:  # a rotation is linear -> backward is inverse rotation
            dq = O.rope_bwd(dq, cos, sin)
            dk = O.rope_bwd(dk, cos, sin)

        d_qf = dq.transpose(0, 2, 1, 3).reshape(B, T, H * cfg.resolved_head_dim)
        d_kf = dk.transpose(0, 2, 1, 3).reshape(B, T, KV * cfg.resolved_head_dim)
        d_vf = dv.transpose(0, 2, 1, 3).reshape(B, T, KV * cfg.resolved_head_dim)
        h = L["h"]

        d_h = _site_backward(p, grads, pre + "attn.q.weight", h, d_qf, train_only) \
            + _site_backward(p, grads, pre + "attn.k.weight", h, d_kf, train_only) \
            + _site_backward(p, grads, pre + "attn.v.weight", h, d_vf, train_only)
        _site_bias_grad(p, grads, pre + "attn.q", d_qf, train_only)
        _site_bias_grad(p, grads, pre + "attn.k", d_kf, train_only)
        _site_bias_grad(p, grads, pre + "attn.v", d_vf, train_only)

        # ---- pre-attention norm ----
        if cfg.norm == "rmsnorm":
            d_xin_a, dw1 = O.rms_norm_bwd(L["aux1"][0], L["aux1"][1], p[pre + "norm1.weight"], d_h)
            if train_only is None or pre + "norm1.weight" in train_only:
                grads[pre + "norm1.weight"] = dw1
        else:
            d_xin_a, dw1, db1 = O.layer_norm_bwd(
                L["aux1"][0], L["aux1"][1], p[pre + "norm1.weight"], d_h
            )
            if train_only is None or pre + "norm1.weight" in train_only:
                grads[pre + "norm1.weight"] = dw1
            if pre + "norm1.bias" in p and (train_only is None or pre + "norm1.bias" in train_only):
                grads[pre + "norm1.bias"] = db1

        # residual: x_mid = x_in + attn_proj
        dx = d_xmid + d_xin_a

    # ---- embeddings ----
    tokens = ctx["tokens"]
    if "lm_head.weight" in p:
        if train_only is None or "wte" in train_only:
            grads["wte"] = O.embedding_bwd(p["wte"].shape, tokens, dx)
    else:
        if train_only is None or "wte" in train_only:
            grads["wte"] = d_wte + O.embedding_bwd(p["wte"].shape, tokens, dx)
    if cfg.pos_encoding == "learned" and (train_only is None or "wpe" in train_only):
        flat_pos = np.tile(np.arange(T), B)
        grads["wpe"] = O.embedding_bwd(p["wpe"].shape, flat_pos, dx)

    return grads


# ---------------------------------------------------------------------------
# Convenience: full loss + gradient in one call
# ---------------------------------------------------------------------------
def loss_and_grad(
    config: ModelConfig,
    params: dict[str, np.ndarray],
    tokens: np.ndarray,
    targets: np.ndarray,
    train_only: set[str] | None = None,
):
    """Cross-entropy loss (mean over positions) and gradients for all params."""
    logits, ctx = forward(config, params, tokens, need_grad=True)
    loss, probs = O.cross_entropy_fwd(logits, targets)
    dlogits = O.cross_entropy_bwd(probs, targets)
    grads = backward(config, params, ctx, dlogits, train_only)
    return float(loss), grads


class Transformer:
    """Thin OO wrapper around :func:`forward` / :func:`backward`."""

    def __init__(self, config: ModelConfig, params: dict[str, np.ndarray]) -> None:
        self.config = config
        self.params = params

    def n_params(self) -> int:
        return sum(int(v.size) for v in self.params.values())

    def forward(self, tokens: np.ndarray, need_grad: bool = True):
        return forward(self.config, self.params, tokens, need_grad=need_grad)

    def loss_and_grad(self, tokens: np.ndarray, targets: np.ndarray):
        return loss_and_grad(self.config, self.params, tokens, targets)

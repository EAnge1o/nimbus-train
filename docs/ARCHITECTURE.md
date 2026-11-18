# ARCHITECTURE — every backward pass, derived by hand

There is no autograd in nimbus-train. Each op below is implemented twice:
once forward, once as the exact vector-Jacobian product. This document
records the derivations so they can be checked against the code — and the
code is checked against them by `tests/test_gradcheck.py` (finite
differences, float64) and `tests/test_torch_parity.py` (real autograd,
1e-9 relative agreement).

Conventions: `y` is an op output, `dy` the upstream gradient, `L` the last
axis size. All ops act along the last axis unless stated otherwise.

---

## Linear — `y = x @ W (+ b)`

The easy one, but note the shapes: weights are stored `(in, out)`, matching
nimbus-infer, so HF-style `(out, in)` checkpoints transpose on import.

```
dx  = dy @ Wᵀ
dW  = xᵀ @ dy        (batch dims flattened)
db  = Σ dy
```

## LayerNorm — `y = x̂ ⊙ w + b`,  `x̂ = (x - μ) / √(σ² + ε)`

With `dyw = dy ⊙ w` (the bias has no gradient path into `x`):

```
dx  = (1/√(σ²+ε)) · ( dyw - mean(dyw) - x̂ · mean(dyw ⊙ x̂) )
dw  = Σ dy ⊙ x̂
db  = Σ dy
```

The three terms are, respectively: the direct path, the gradient through
the mean, and the gradient through the variance. Derivation sketch: write
`x̂ = A(x)·(x - μ)` with `A = 1/√(σ²+ε)`; differentiate `y = A·(x-μ)·w + b`
w.r.t. `x` and simplify the two rank-1 projections into the means above.

## RMSNorm — `y = (x / √(mean(x²) + ε)) ⊙ w`

Same structure, one fewer projection (no mean subtraction):

```
dx  = (1/√(ms+ε)) · ( dyw - x̂ · mean(dyw ⊙ x̂) )
dw  = Σ dy ⊙ x̂
```

## GELU (tanh approximation) — `y = 0.5x(1 + t)`,  `t = tanh(c(x + 0.044715x³))`

```
dt/dx = c · (1 + 3·0.044715·x²)
dy/dx = 0.5(1 + t) + 0.5 · x · (1 - t²) · dt/dx
```

The backward stores only `t` (the tanh output) — `x` is already kept by the
caller for the weight gradients.

## SiLU — `y = x · s`,  `s = σ(x)`

```
dy/dx = s + x·s·(1 - s) = s · (1 + x(1 - s))
```

## Softmax — `p = softmax(z)` (last axis)

The Jacobian is `diag(p) - ppᵀ`; its product with `dy` collapses into:

```
dz = p ⊙ (dy - Σ(dy ⊙ p))
```

This op is **destructive in our code**: attention scores and logits are
never needed after softmax, so the probability matrix is written in place
(one fewer 67 MB array per layer per step).

## Cross-entropy fused with softmax — `loss = -mean(log p[target])`

```
dlogits = (p - onehot(target)) / n
```

`n` is the number of positions. Implemented as two in-place passes over the
probability matrix (subtract 1 at targets, divide by n) — no zeros-allocation
for a fresh gradient tensor.

## Causal attention — scores → mask → softmax → value mix

Per head, with `S = (q @ kᵀ) · scale`, `p = softmax(S + mask)`, `out = p @ v`:

```
dv      = pᵀ @ dout
dprobs  = dout @ vᵀ
dS      = p ⊙ (dprobs - Σ(dprobs ⊙ p))        # softmax bwd, in place
dq      = (dS · scale) @ k
dk      = (dS · scale)ᵀ @ q
```

Ordering matters for the in-place variant: `dv` must be computed *before*
`dS` overwrites `p`. The additive mask (`0` / `-1e9`, cached per sequence
length) contributes nothing to the gradient — masked positions have `p = 0`,
so `dS` is exactly zero there and no gradient flows into masked scores.

## GQA (grouped-query attention)

`k, v` have `n_kv` heads and are repeated `g = n_heads / n_kv` times. The
repeat is a linear broadcast, so its backward sums over the group:

```
dk_kv = Σ_{i=0..g-1} dk[i·g:(i+1)·g]        (reshape (kv, g, ...) → sum axis 1)
```

## RoPE — `y₁ = x₁c - x₂s`,  `y₂ = x₁s + x₂c`

The rotation matrix `[[c, -s], [s, c]]` is orthogonal, so the backward is
the inverse rotation, i.e. its transpose:

```
dx₁ = dy₁·c + dy₂·s
dx₂ = -dy₁·s + dy₂·c
```

No tables are stored for the backward — cos/sin are re-sliced from the
precomputed tables in `ctx`.

## Embedding — `y[i] = W[ids[i]]`

Scatter-add: `np.add.at(dW, ids_flat, dy_flat)`. Repeated ids (frequent
tokens) accumulate — the finite-difference test uses `id = 7` twice for
exactly this reason.

## Weight tying

With tying, `logits = x_f @ wteᵀ`, so `wte` receives **two** gradient
contributions that are summed:

```
d_wte += dlᵀ @ x_f          # from the output head
d_wte += scatter_add(dx_emb) # from the input embedding
```

## The residual stream

Each block computes `x ← x + attn(norm₁(x))` then `x ← x + mlp(norm₂(x))`.
Backward unrolls the three "copies" of each intermediate explicitly:

```
d(x_mid) = d(block_out)              # through x = x_mid + mlp_out
         + norm₂_bwd(mlp_bwd)        # through the pre-MLP norm
d(x_in)  = d(x_mid)                  # through x_mid = x_in + attn_proj
         + norm₁_bwd(attn_bwd)       # through the pre-attention norm
```

---

## LoRA — `y = x @ W + (x @ A) @ B`

Adapters are ordinary dict entries (`W.lora_A` `(in, r)`, `W.lora_B`
`(r, out)`), so the model code treats them as an extra low-rank path:

```
dx = dy @ Wᵀ + (dy @ Bᵀ) @ Aᵀ
dA = xᵀ @ (dy @ Bᵀ)
dB = (x @ A)ᵀ @ dy
```

`B = 0` at init ⇒ the adapter is a no-op (asserted bit-identical in tests),
and the `alpha/r` scaling is absorbed into `A` at attachment. When the base
is frozen, `backward(..., train_only=lora_keys)` **skips the `xᵀ @ dy`
matmuls of frozen weights entirely** — on CPU that is roughly half the
gradient cost, not just fewer updates.

## DPO — policy gradients without a reward model

For `loss = -log σ(z)` with
`z = β·[(logπ(y_w) - logπ_ref(y_w)) - (logπ(y_l) - logπ_ref(y_l))]`:

```
∂loss/∂z  = -(1 - σ(z)) / n_pairs
∂loss/∂logit_t = coef · (onehot(y_t) - p_t)     # per masked position
```

with `coef = ∓β(1-σ(z))/n_pairs` for chosen/rejected rows. The frozen
reference model is run with `need_grad=False`; with LoRA the backward only
touches adapters.

---

## CPU engineering

A NumPy training step has two cost classes:

* **BLAS matmuls** — already parallel. An i7-12700H reaches ~330 GFLOPS in
  float32; the lm-head matmul alone is ~33 GFLOP per step.
* **Elementwise loops** (`np.exp`, norms, GELU) — *single-threaded*, and
  `exp` is compute-bound at ~tens of cycles per element. Before threading,
  one flagship step was ~8.2 s; after, ~4.3 s at the same bit-exact results.

Three measures, all inside `ops.py`:

1. **Threaded fan-out.** NumPy ufuncs release the GIL on large arrays, so
   hot ops split their leading axis across a small `ThreadPoolExecutor`
   (`_map_chunks`). The chunk math is identical, so results are
   bit-identical to the serial path — the gradient tests run through the
   same code. Threshold: below ~1.5 M elements the serial path wins.
2. **In-place evaluation.** `softmax_fwd` (max-shift, `exp`, divide) and the
   additive causal mask write into their input; CE's `(p - onehot)/n`
   writes into the probability matrix. Saves several 67–268 MB
   allocations per step.
3. **Cached masks.** The additive causal mask is built once per
   `(seq_len, dtype)`.

`NIMBUS_NUM_WORKERS` caps the pool (tests and CI set it low).

---

## Checkpoint formats

A **run directory** is the training-side format:

```
run/
  config.json          ModelConfig (identical schema to nimbus-infer)
  train_config.json    hyperparameters + current step
  params.npz           weights (incl. any LoRA adapters)
  optimizer.npz        AdamW moments (resume support)
  log.csv              step, train_loss, val_loss, lr, tok/s, elapsed
  tokenizer/           vocab.json + merges.txt (GPT-2 format)
```

An **export directory** is what nimbus-infer consumes:

```
ckpt/
  config.json          ModelConfig
  weights.safetensors  hand-written safetensors container, fp32, (in, out)
  tokenizer/           vocab.json + merges.txt
```

`export.py` writes the safetensors header by hand (uint64 length + JSON +
raw buffers) and re-reads it to assert logit parity with the training
model — the export is verified, not assumed.

"""nimbus-train: train transformers from scratch in pure NumPy.

Hand-written forward *and* backward passes -- no autograd, no framework,
numpy as the only dependency. Exports checkpoints that nimbus-infer loads
for quantized, cached inference.

Pipeline: train-tokenizer -> prepare-data -> pretrain -> sft/dpo -> export
"""
from .config import ModelConfig
from .model import Transformer, backward, forward, init_params, loss_and_grad

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "Transformer",
    "init_params",
    "forward",
    "backward",
    "loss_and_grad",
    "__version__",
]

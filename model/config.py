from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a LLaMA-style decoder-only Transformer.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        n_layers: Number of Transformer decoder blocks.
        d_model: Hidden dimension (embedding size).
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value attention heads (GQA when < n_heads).
        d_ff: Intermediate dimension of the SwiGLU MLP.
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
        rope_theta: Base frequency for Rotary Position Embeddings (RoPE).
        tie_word_embeddings: Whether lm_head shares weights with embed_tokens.
    """

    vocab_size: int = 512
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_json(cls, path: str | Path, **overrides: Any) -> "ModelConfig":
        """Load a ModelConfig from a JSON file in `configs/`."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.update(overrides)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LayerKVCache(NamedTuple):
    """Static-shape KV cache for a single Transformer layer.

    Using a NamedTuple automatically registers LayerKVCache as a JAX PyTree,
    so it can be passed freely through `jax.jit` and `jax.lax.scan`.

    Attributes:
        k: Key cache of shape [B, max_seq_len, n_kv_heads, head_dim].
        v: Value cache of shape [B, max_seq_len, n_kv_heads, head_dim].
    """

    k: jax.Array
    v: jax.Array


# Full model KV cache is a tuple of `n_layers` LayerKVCache entries (also a PyTree).
ModelKVCache = tuple[LayerKVCache, ...]

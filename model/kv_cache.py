"""Assignment 1.2: Static-Shape KV Cache Initialization & Functional Update."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import LayerKVCache, ModelConfig, ModelKVCache


def init_kv_cache(
    config: ModelConfig,
    batch_size: int,
    max_seq_len: int,
    dtype: jnp.dtype = jnp.float32,
) -> ModelKVCache:
    """Allocate a zero-initialized, static-shape KV cache for all layers.

    Args:
        config: ModelConfig specifying n_layers, n_kv_heads, and head_dim.
        batch_size: Static batch size B.
        max_seq_len: Maximum total sequence length (prompt + generated tokens).
        dtype: Data type for cached keys and values (e.g., bfloat16 on TPU).

    Returns:
        Tuple of `config.n_layers` LayerKVCache entries, each with `k` and `v`
        of shape [B, max_seq_len, n_kv_heads, head_dim].
    """
    # BEGIN Assignment 1.2
    raise NotImplementedError("Implement Assignment 1.2")
    # END Assignment 1.2


def update_kv_cache(
    layer_cache: LayerKVCache,
    k_new: jax.Array,
    v_new: jax.Array,
    cache_index: jax.Array | int,
) -> LayerKVCache:
    """Write new K/V tensors into a static-shape layer cache at `cache_index`.

    Uses `jax.lax.dynamic_update_slice` so XLA compiles a single static-shape
    program for both prefill (`T = P`, `cache_index = 0`) and decode
    (`T = 1`, `cache_index = P + step`) without recompilation.

    Args:
        layer_cache: Existing LayerKVCache with k, v of shape [B, S, n_kv_heads, head_dim].
        k_new: New keys of shape [B, T, n_kv_heads, head_dim].
        v_new: New values of shape [B, T, n_kv_heads, head_dim].
        cache_index: Scalar int32 start offset along the sequence axis (axis 1).

    Returns:
        Updated LayerKVCache with the same static shapes.
    """
    # BEGIN Assignment 1.2
    raise NotImplementedError("Implement Assignment 1.2")
    # END Assignment 1.2

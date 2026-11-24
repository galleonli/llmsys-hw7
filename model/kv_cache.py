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
        zero-initialized to shape [B, max_seq_len, n_kv_heads, head_dim] and `dtype`.
    """
    # BEGIN Part 2.1
    raise NotImplementedError("Implement Part 2.1")
    # END Part 2.1


def update_kv_cache(
    layer_cache: LayerKVCache,
    k_new: jax.Array,
    v_new: jax.Array,
    cache_index: jax.Array | int,
) -> LayerKVCache:
    """Write new K/V tensors into a static-shape layer cache at `cache_index`.

    Args:
        layer_cache: Existing LayerKVCache with k, v of shape [B, S, n_kv_heads, head_dim].
        k_new: New keys of shape [B, T, n_kv_heads, head_dim].
        v_new: New values of shape [B, T, n_kv_heads, head_dim].
        cache_index: Scalar int32 start offset along the sequence axis (axis 1).

    Returns:
        Updated LayerKVCache with the same static shapes.

    Constraints:
        - Functionally write into slots [cache_index, cache_index + T) without mutating
          the input cache, casting `k_new`/`v_new` to match the cache dtype.
        - Must trace only once under `@jax.jit` across varying `cache_index` values
          (e.g., via `jax.lax.dynamic_update_slice`).
    """
    # BEGIN Part 2.2
    raise NotImplementedError("Implement Part 2.2")
    # END Part 2.2

"""Assignment 1.3: Grouped-Query Attention & Causal/Padding Attention Mask."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_attention_mask(
    cache_valid_mask: jax.Array,
    cache_index: jax.Array | int,
    query_len: int,
) -> jax.Array:
    """Build a combined causal + valid-token boolean attention mask.

    Args:
        cache_valid_mask: Boolean mask of shape [B, max_seq_len] indicating which
            KV cache slots hold valid (non-pad, already-written) tokens.
        cache_index: Scalar int32 starting cache slot of the current queries.
        query_len: Static Python int T (number of query tokens in this step).

    Returns:
        Boolean attention mask of shape [B, 1, query_len, max_seq_len] where
        True means the query is allowed to attend to the key slot.
    """
    # BEGIN Assignment 1.3
    raise NotImplementedError("Implement Assignment 1.3")
    # END Assignment 1.3


def gqa_attention(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    attn_mask: jax.Array,
) -> jax.Array:
    """Grouped-Query Attention over a static-shape KV cache.

    Args:
        q: Query tensor of shape [B, T, n_heads, head_dim].
        k_cache: Key cache of shape [B, max_seq_len, n_kv_heads, head_dim].
        v_cache: Value cache of shape [B, max_seq_len, n_kv_heads, head_dim].
        attn_mask: Boolean mask of shape [B, 1, T, max_seq_len] (True = attend).

    Returns:
        Attention output of shape [B, T, n_heads, head_dim] with dtype `q.dtype`.
    """
    # BEGIN Assignment 1.3
    raise NotImplementedError("Implement Assignment 1.3")
    # END Assignment 1.3

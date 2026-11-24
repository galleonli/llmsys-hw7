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
        entry (b, 0, t, s) is True iff `cache_valid_mask[b, s]` is True and
        `s <= cache_index + t`.
    """
    # BEGIN Part 3.1
    raise NotImplementedError("Implement Part 3.1")
    # END Part 3.1


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

    Constraints:
        - Group query heads so each KV head serves `n_heads // n_kv_heads` query heads
          without copying `k_cache` or `v_cache`.
        - Compute attention scores and softmax in float32.
        - Masked-out slots must have zero effect on outputs, and all-masked query rows
          (from left-padding) must produce finite values (no NaN/Inf).
    """
    # BEGIN Part 3.2
    raise NotImplementedError("Implement Part 3.2")
    # END Part 3.2

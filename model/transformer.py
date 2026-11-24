"""Assignment 4: Transformer Block, Prefill, and Single-Step Decode."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .attention import gqa_attention, make_attention_mask
from .config import LayerKVCache, ModelConfig, ModelKVCache
from .kv_cache import init_kv_cache, update_kv_cache
from .layers import apply_rope, rms_norm, swiglu_mlp


def transformer_block(
    x: jax.Array,
    layer_params: dict[str, jax.Array],
    layer_cache: LayerKVCache,
    positions: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    cache_index: jax.Array | int,
    attn_mask: jax.Array,
    config: ModelConfig,
) -> tuple[jax.Array, LayerKVCache]:
    """Run one pre-norm LLaMA Transformer block with KV cache update.

    Args:
        x: Hidden states of shape [B, T, d_model].
        layer_params: Dictionary of weights for this layer.
        layer_cache: Existing LayerKVCache for this layer.
        positions: Token RoPE positions of shape [B, T] (int32).
        cos, sin: RoPE tables of shape [max_seq_len, head_dim // 2].
        cache_index: Scalar int32 start offset in the KV cache.
        attn_mask: Boolean mask of shape [B, 1, T, max_seq_len].
        config: ModelConfig.

    Returns:
        (out, new_layer_cache) where `out` has shape [B, T, d_model].
    """
    # BEGIN Assignment 4
    raise NotImplementedError("Implement Assignment 4")
    # END Assignment 4


def prefill(
    params: dict[str, Any],
    prompt_ids: jax.Array,
    prompt_mask: jax.Array,
    config: ModelConfig,
    max_seq_len: int,
    cos: jax.Array,
    sin: jax.Array,
) -> tuple[jax.Array, ModelKVCache, jax.Array, jax.Array]:
    """Run the compute-bound prefill phase over a left-padded prompt batch.

    Args:
        params: Model parameter PyTree.
        prompt_ids: Left-padded prompt token IDs of shape [B, P] (int32).
        prompt_mask: Boolean/int mask of shape [B, P] (True/1 for real tokens).
        config: ModelConfig.
        max_seq_len: Static maximum sequence length for the KV cache.
        cos, sin: RoPE tables of shape [max_seq_len, head_dim // 2].

    Returns:
        logits: Output logits of shape [B, P, vocab_size].
        cache: Populated ModelKVCache with prompts written to slots [0, P).
        cache_valid_mask: Boolean mask of shape [B, max_seq_len].
        next_positions: True RoPE position for the first decode token, shape [B, 1] (int32).

    Constraints:
        - Support left-padded prompts: real tokens of sequence b receive RoPE positions
          0, 1, ..., L_b - 1 (pad tokens receive 0), and `next_positions[b, 0] == L_b`.
        - Support both tied (`embed_tokens.T`) and untied (`lm_head`) output projections.
    """
    # BEGIN Assignment 4
    raise NotImplementedError("Implement Assignment 4")
    # END Assignment 4


def decode_step(
    params: dict[str, Any],
    token_ids: jax.Array,
    positions: jax.Array,
    cache: ModelKVCache,
    cache_valid_mask: jax.Array,
    cache_index: jax.Array | int,
    cos: jax.Array,
    sin: jax.Array,
    config: ModelConfig,
) -> tuple[jax.Array, ModelKVCache, jax.Array]:
    """Run one memory-bandwidth-bound autoregressive decode step (T = 1).

    Args:
        params: Model parameter PyTree.
        token_ids: Current input token IDs of shape [B, 1] (int32).
        positions: Per-sequence true RoPE positions of shape [B, 1] (int32).
        cache: Current ModelKVCache.
        cache_valid_mask: Boolean mask of shape [B, max_seq_len].
        cache_index: Scalar int32 write slot in the KV cache (P + step).
        cos, sin: RoPE tables of shape [max_seq_len, head_dim // 2].
        config: ModelConfig.

    Returns:
        logits: Next-token logits of shape [B, 1, vocab_size].
        new_cache: Updated ModelKVCache.
        new_cache_valid_mask: Updated boolean mask of shape [B, max_seq_len].

    Constraints:
        Marks column `cache_index` valid, writes single-token K/V at `cache_index`,
        and must trace only once under `@jax.jit` across consecutive decode steps.
    """
    # BEGIN Assignment 4
    raise NotImplementedError("Implement Assignment 4")
    # END Assignment 4

"""Assignment 5: Sampling & JIT-Compiled Batched Generation with jax.lax.scan."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .layers import precompute_rope_freqs
from .transformer import decode_step, prefill


def sample_next_token(
    logits: jax.Array,
    rng_key: jax.Array,
    temperature: float = 0.0,
    top_k: int = 0,
) -> jax.Array:
    """Sample the next token ID for each sequence in the batch.

    Args:
        logits: Unnormalized logits of shape [B, vocab_size].
        rng_key: JAX PRNG key.
        temperature: Sampling temperature. If <= 0.0, perform greedy argmax.
        top_k: If > 0 (and temperature > 0), restrict sampling to the top-k logits.

    Returns:
        Sampled token IDs of shape [B] (int32).
    """
    # BEGIN Assignment 5
    raise NotImplementedError("Implement Assignment 5")
    # END Assignment 5


@functools.partial(
    jax.jit,
    static_argnames=("config", "max_new_tokens", "temperature", "top_k"),
)
def generate(
    params: dict[str, Any],
    prompt_ids: jax.Array,
    prompt_mask: jax.Array,
    config: ModelConfig,
    max_new_tokens: int,
    rng_key: jax.Array,
    temperature: float = 0.0,
    top_k: int = 0,
) -> jax.Array:
    """End-to-end JIT-compiled prefill + jax.lax.scan batched decode.

    Args:
        params: Model parameter PyTree.
        prompt_ids: Left-padded prompt token IDs of shape [B, P] (int32).
        prompt_mask: Boolean/int mask of shape [B, P] (True/1 for real tokens).
        config: ModelConfig (static).
        max_new_tokens: Number of new tokens to generate (static, >= 1).
        rng_key: JAX PRNG key.
        temperature: Sampling temperature (static; <= 0.0 for greedy).
        top_k: Top-k cutoff (static; 0 disables top-k filtering).

    Returns:
        Generated token IDs of shape [B, max_new_tokens] (int32), excluding the prompt.
    """
    # BEGIN Assignment 5
    raise NotImplementedError("Implement Assignment 5")
    # END Assignment 5

"""Assignment 1.1: Core LLaMA Building Blocks (RMSNorm, RoPE, SwiGLU)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def precompute_rope_freqs(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[jax.Array, jax.Array]:
    """Precompute RoPE cosine and sine tables in float32.

    Returns:
        (cos, sin), each of shape [max_seq_len, head_dim // 2] in float32.
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) * 2.0 / head_dim))
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, inv_freq)  # [max_seq_len, head_dim // 2]
    return jnp.cos(angles), jnp.sin(angles)


def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Root Mean Square Layer Normalization.

    Args:
        x: Input activations of shape [..., d_model].
        weight: Scale parameter of shape [d_model].
        eps: Small constant for numerical stability.

    Returns:
        Normalized activations of shape [..., d_model] with the same dtype as `x`.
    """
    # BEGIN Assignment 1.1
    raise NotImplementedError("Implement Assignment 1.1")
    # END Assignment 1.1


def apply_rope(
    x: jax.Array,
    positions: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply Rotary Position Embeddings (rotate-half convention) to Q or K.

    Args:
        x: Tensor of shape [B, T, H, head_dim].
        positions: Integer token positions of shape [B, T].
        cos: Precomputed cosines of shape [max_seq_len, head_dim // 2].
        sin: Precomputed sines of shape [max_seq_len, head_dim // 2].

    Returns:
        Rotated tensor of shape [B, T, H, head_dim] with the same dtype as `x`.
    """
    # BEGIN Assignment 1.1
    raise NotImplementedError("Implement Assignment 1.1")
    # END Assignment 1.1


def swiglu_mlp(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
) -> jax.Array:
    """SwiGLU feed-forward network: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down.

    Args:
        x: Input activations of shape [B, T, d_model].
        w_gate: Gate projection of shape [d_model, d_ff].
        w_up: Up projection of shape [d_model, d_ff].
        w_down: Down projection of shape [d_ff, d_model].

    Returns:
        Output activations of shape [B, T, d_model].
    """
    # BEGIN Assignment 1.1
    raise NotImplementedError("Implement Assignment 1.1")
    # END Assignment 1.1

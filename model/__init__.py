"""LLaMA-style pure-JAX inference package for CMU 11-868."""

from .attention import gqa_attention, make_attention_mask
from .config import LayerKVCache, ModelConfig, ModelKVCache
from .generate import generate, sample_next_token
from .kv_cache import init_kv_cache, update_kv_cache
from .layers import apply_rope, precompute_rope_freqs, rms_norm, swiglu_mlp
from .transformer import decode_step, prefill, transformer_block
from .utils import (
    estimate_tpu_v5e_roofline,
    init_model_params,
    left_pad_prompts,
    load_hf_llama_weights,
)

__all__ = [
    "LayerKVCache",
    "ModelConfig",
    "ModelKVCache",
    "apply_rope",
    "decode_step",
    "estimate_tpu_v5e_roofline",
    "generate",
    "gqa_attention",
    "init_kv_cache",
    "init_model_params",
    "left_pad_prompts",
    "load_hf_llama_weights",
    "make_attention_mask",
    "precompute_rope_freqs",
    "prefill",
    "rms_norm",
    "sample_next_token",
    "swiglu_mlp",
    "transformer_block",
    "update_kv_cache",
]

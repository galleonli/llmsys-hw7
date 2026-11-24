# CMU 11-868: LLM Inference with JAX on TPU v5e

Implementation-first homework on single-chip LLM inference in pure JAX (LLaMA-style decoder with RMSNorm, RoPE, GQA, SwiGLU, static-shape KV caching, and `jax.lax.scan` batched variable-length decoding) targeting one Google Cloud TPU v5e.

## Repository Layout
```text
.
├── configs/
│   ├── model_tiny.json             # Default small model config
│   ├── model_test.json             # Unit-test model config
│   └── model_benchmark.json        # Roofline benchmark model config
├── data/
│   └── test_data.npz               # Precomputed golden test fixtures
├── figures/
│   ├── llama_block.png             # Pre-Norm LLaMA Transformer block diagram
│   ├── kv_cache_left_pad.png       # Left-padded static KV cache layout diagram
│   └── gqa_grouping.png            # Zero-copy GQA head-grouping diagram
├── model/
│   ├── __init__.py                 # Public package exports
│   ├── config.py                   # ModelConfig (JSON loader) & LayerKVCache / ModelKVCache
│   ├── layers.py                   # Part 1 (1.1–1.3): rms_norm, apply_rope, swiglu_mlp
│   ├── kv_cache.py                 # Part 2 (2.1–2.2): init_kv_cache, update_kv_cache
│   ├── attention.py                # Part 3 (3.1–3.2): make_attention_mask, gqa_attention
│   ├── transformer.py              # Part 4 (4.1–4.3): transformer_block, prefill, decode_step
│   ├── generate.py                 # Part 5 (5.1–5.2): sample_next_token, generate
│   └── utils.py                    # Provided helpers (init, left-pad, roofline, HF loader)
├── tests/
│   ├── __init__.py
│   └── test_inference.py           # Pytest correctness & TPU speedup test suite
├── hw_inference.ipynb              # Student notebook (specs, illustrations, checks, grading)
└── README.md
```

## Homework Overview
| Part | Module | Functions | Core Concepts |
| :--- | :--- | :--- | :--- |
| **Part 1** (`1.1`–`1.3`) | `model/layers.py` | `rms_norm`, `apply_rope`, `swiglu_mlp` | Pure JAX ops, `float32` accumulation, rotate-half RoPE |
| **Part 2** (`2.1`–`2.2`) | `model/kv_cache.py` | `init_kv_cache`, `update_kv_cache` | PyTree `NamedTuple` caches, static shapes, `jax.lax.dynamic_update_slice` |
| **Part 3** (`3.1`–`3.2`) | `model/attention.py` | `make_attention_mask`, `gqa_attention` | Causal + valid-slot masking, zero-copy GQA grouping, NaN-safe masking |
| **Part 4** (`4.1`–`4.3`) | `model/transformer.py` | `transformer_block`, `prefill`, `decode_step` | Prefill vs decode, single-trace XLA verification |
| **Part 5** (`5.1`–`5.2`) | `model/generate.py` | `sample_next_token`, `generate` | Explicit PRNG keys, top-k sampling, `@jax.jit` + `jax.lax.scan`, left-padded batching |

## Quickstart
Open `hw_inference.ipynb` in a **TPU v5e** runtime, implement each section inside `# BEGIN Part X.Y` ... `# END Part X.Y` in `model/`, and run the per-part test cells and the final grading cell — or run `pytest` directly:
```bash
pytest -v tests/test_inference.py -k part_1   # single part
pytest -v tests/test_inference.py             # full suite (TPU v5e)
```

## Background References
- Google Developers Blog: [Train a GPT-2 Model with JAX on TPU](https://developers.googleblog.com/train-gpt2-model-with-jax-on-tpu/)
- Google DeepMind Scaling Book: [Part 7: All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)

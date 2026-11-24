#!/usr/bin/env python3
"""Standalone student-safe grading & performance test suite for CMU 11-868.

Compares student outputs directly against precomputed golden fixtures in
`data/test_data.npz` without exposing any reference solution code.

Usage:
    python3 tests/test_inference.py
    python3 tests/test_inference.py model --benchmark
    python3 tests/test_inference.py path/to/student_release/model --benchmark
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "model"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "test_data.npz"
CONFIGS_DIR = REPO_ROOT / "configs"


# =============================================================================
# Module & Fixture Loaders
# =============================================================================


def load_submission(submission_path: str | Path) -> ModuleType:
    """Load a submission package directory (e.g. `model/`) or single `.py` file."""
    path = Path(submission_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Submission path not found: {path}")

    if path.is_dir():
        init_py = path / "__init__.py"
        if not init_py.exists():
            raise FileNotFoundError(f"Package directory missing __init__.py: {path}")
        pkg_name = "submitted_model"
        for name in list(sys.modules):
            if name == pkg_name or name.startswith(f"{pkg_name}."):
                del sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            str(init_py),
            submodule_search_locations=[str(path)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load package from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = mod
        spec.loader.exec_module(mod)
        return mod

    spec = importlib.util.spec_from_file_location("submitted_model", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submitted_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model_from_fixture(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    n_layers: int,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    """Reconstruct a JAX parameter PyTree from saved golden weights."""
    layer_keys = ("attn_norm", "w_q", "w_k", "w_v", "w_o", "mlp_norm", "w_gate", "w_up", "w_down")
    layers = tuple(
        {k: jnp.asarray(data[f"{prefix}_l{i}_{k}"]) for k in layer_keys}
        for i in range(n_layers)
    )
    params: dict[str, Any] = {
        "embed_tokens": jnp.asarray(data[f"{prefix}_embed_tokens"]),
        "layers": layers,
        "final_norm": jnp.asarray(data[f"{prefix}_final_norm"]),
    }
    if not tie_word_embeddings:
        params["lm_head"] = jnp.asarray(data[f"{prefix}_lm_head"])
    return params


# =============================================================================
# Correctness & Performance Tests (Golden Data Comparison Only)
# =============================================================================


def test_assignment_1_1(mod: ModuleType, data: np.lib.npyio.NpzFile) -> None:
    # 1. rms_norm
    x = jnp.asarray(data["a11_rms_x"])
    w = jnp.asarray(data["a11_rms_w"])
    out = mod.rms_norm(x, w, eps=1e-6)
    assert out.shape == x.shape, f"rms_norm shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == x.dtype, f"rms_norm dtype mismatch: {out.dtype} vs {x.dtype}"
    np.testing.assert_allclose(np.asarray(out), data["a11_rms_expected"], rtol=1e-5, atol=1e-5)

    # bfloat16 dtype preservation
    out_bf16 = mod.rms_norm(x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), eps=1e-6)
    assert out_bf16.dtype == jnp.bfloat16, f"rms_norm should preserve bfloat16, got {out_bf16.dtype}"

    # 2. apply_rope
    q = jnp.asarray(data["a11_rope_q"])
    pos = jnp.asarray(data["a11_rope_pos"])
    cos = jnp.asarray(data["a11_rope_cos"])
    sin = jnp.asarray(data["a11_rope_sin"])
    q_rot = mod.apply_rope(q, pos, cos, sin)
    assert q_rot.shape == q.shape, f"apply_rope shape mismatch: {q_rot.shape}"
    assert q_rot.dtype == q.dtype
    np.testing.assert_allclose(np.asarray(q_rot), data["a11_rope_expected"], rtol=1e-5, atol=1e-5)

    # Orthogonality (norm preservation) & relative-position dot-product invariance
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(q_rot, axis=-1)),
        np.asarray(jnp.linalg.norm(q, axis=-1)),
        rtol=1e-5,
        atol=1e-5,
    )
    q0, k0 = q[:1, :1], q[1:2, :1]
    dot_a = jnp.sum(
        mod.apply_rope(q0, jnp.array([[5]]), cos, sin)
        * mod.apply_rope(k0, jnp.array([[2]]), cos, sin),
        axis=-1,
    )
    dot_b = jnp.sum(
        mod.apply_rope(q0, jnp.array([[15]]), cos, sin)
        * mod.apply_rope(k0, jnp.array([[12]]), cos, sin),
        axis=-1,
    )
    np.testing.assert_allclose(np.asarray(dot_a), np.asarray(dot_b), rtol=1e-4, atol=1e-4)

    # 3. swiglu_mlp
    mlp_out = mod.swiglu_mlp(
        jnp.asarray(data["a11_mlp_x"]),
        jnp.asarray(data["a11_mlp_wg"]),
        jnp.asarray(data["a11_mlp_wu"]),
        jnp.asarray(data["a11_mlp_wd"]),
    )
    assert mlp_out.shape == data["a11_mlp_expected"].shape
    np.testing.assert_allclose(np.asarray(mlp_out), data["a11_mlp_expected"], rtol=1e-4, atol=1e-4)


def test_assignment_1_2(mod: ModuleType, _data: np.lib.npyio.NpzFile) -> None:
    cfg = mod.ModelConfig(n_layers=3, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
    bsz, max_len = 2, 16
    cache = mod.init_kv_cache(cfg, bsz, max_len, dtype=jnp.float32)

    assert len(cache) == cfg.n_layers, f"Expected {cfg.n_layers} layers, got {len(cache)}"
    for layer_cache in cache:
        assert layer_cache.k.shape == (bsz, max_len, cfg.n_kv_heads, cfg.head_dim)
        assert layer_cache.v.shape == (bsz, max_len, cfg.n_kv_heads, cfg.head_dim)
        assert jnp.all(layer_cache.k == 0.0) and jnp.all(layer_cache.v == 0.0)

    # Prefill slice write at index 0 (T = 4)
    key = jax.random.PRNGKey(12)
    k_pre = jax.random.normal(key, (bsz, 4, cfg.n_kv_heads, cfg.head_dim))
    v_pre = jax.random.normal(jax.random.fold_in(key, 1), (bsz, 4, cfg.n_kv_heads, cfg.head_dim))
    lc1 = mod.update_kv_cache(cache[0], k_pre, v_pre, 0)
    np.testing.assert_allclose(np.asarray(lc1.k[:, :4]), np.asarray(k_pre))
    np.testing.assert_allclose(np.asarray(lc1.v[:, :4]), np.asarray(v_pre))
    assert jnp.all(lc1.k[:, 4:] == 0.0)
    assert jnp.all(cache[0].k == 0.0), "Original KV cache must remain immutable"

    # Sequential decode writes + XLA single-trace verification
    trace_count = 0

    @jax.jit
    def jitted_update(lc, kn, vn, idx):
        nonlocal trace_count
        trace_count += 1
        return mod.update_kv_cache(lc, kn, vn, idx)

    lc = lc1
    for step in range(4):
        k_tok = jnp.full((bsz, 1, cfg.n_kv_heads, cfg.head_dim), float(step + 1))
        v_tok = jnp.full((bsz, 1, cfg.n_kv_heads, cfg.head_dim), float(-(step + 1)))
        lc = jitted_update(lc, k_tok, v_tok, jnp.asarray(4 + step, dtype=jnp.int32))

    assert trace_count == 1, f"Expected 1 XLA trace for update_kv_cache, got {trace_count}"
    for step in range(4):
        np.testing.assert_allclose(np.asarray(lc.k[:, 4 + step]), float(step + 1))
        np.testing.assert_allclose(np.asarray(lc.v[:, 4 + step]), float(-(step + 1)))


def test_assignment_1_3(mod: ModuleType, data: np.lib.npyio.NpzFile) -> None:
    # 1. make_attention_mask (prefill & decode)
    mask_pre = mod.make_attention_mask(jnp.asarray(data["a13_valid_pre"]), 0, 4)
    assert mask_pre.shape == data["a13_mask_pre_expected"].shape
    assert mask_pre.dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(mask_pre), data["a13_mask_pre_expected"])

    mask_dec = mod.make_attention_mask(jnp.asarray(data["a13_valid_dec"]), 4, 1)
    assert mask_dec.shape == data["a13_mask_dec_expected"].shape
    np.testing.assert_array_equal(np.asarray(mask_dec), data["a13_mask_dec_expected"])

    # 2. gqa_attention against golden output + masked-slot garbage invariance
    q = jnp.asarray(data["a13_gqa_q"])
    k_cache = jnp.asarray(data["a13_gqa_k"])
    v_cache = jnp.asarray(data["a13_gqa_v"])
    mask = jnp.asarray(data["a13_mask_pre_expected"])

    out = mod.gqa_attention(q, k_cache, v_cache, mask)
    assert out.shape == q.shape
    assert jnp.all(jnp.isfinite(out)), "gqa_attention produced NaN/Inf (check all-masked left-pad rows!)"
    np.testing.assert_allclose(np.asarray(out), data["a13_gqa_expected"], rtol=1e-4, atol=1e-4)

    # Inject large garbage into masked-out slots (4..7) to verify mask invariance
    out_dirty = mod.gqa_attention(q, k_cache.at[:, 4:].set(1e5), v_cache.at[:, 4:].set(-1e5), mask)
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(out_dirty[0]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(out[1, 2:]), np.asarray(out_dirty[1, 2:]), rtol=1e-4, atol=1e-4)


def test_assignment_1_4(mod: ModuleType, data: np.lib.npyio.NpzFile) -> None:
    cos = jnp.asarray(data["m_cos"])
    sin = jnp.asarray(data["m_sin"])
    prompt_ids = jnp.asarray(data["a14_prompt_ids"])
    prompt_mask = jnp.asarray(data["a14_prompt_mask"])
    prompt_len = prompt_ids.shape[1]
    max_seq_len = 20

    for tag, tie in (("tied", True), ("untied", False)):
        cfg = mod.ModelConfig.from_json(CONFIGS_DIR / "model_test.json", tie_word_embeddings=tie)
        params = load_model_from_fixture(data, f"m_{tag}", cfg.n_layers, tie)

        logits_pre, cache, valid_mask, next_pos = mod.prefill(
            params, prompt_ids, prompt_mask, cfg, max_seq_len, cos, sin
        )
        expected_pre = data[f"a14_{tag}_prefill_logits"]
        assert logits_pre.shape == expected_pre.shape
        np.testing.assert_allclose(np.asarray(logits_pre[0]), expected_pre[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(logits_pre[1, 2:]), expected_pre[1, 2:], rtol=1e-4, atol=1e-4)

        # Single-trace check on jitted decode_step + 5 sequential steps against golden logits
        trace_count = 0

        @jax.jit
        def jitted_decode(tok, pos, c, vm, idx):
            nonlocal trace_count
            trace_count += 1
            return mod.decode_step(params, tok, pos, c, vm, idx, cos, sin, cfg)

        cur_pos = next_pos
        dec_inputs = data[f"a14_{tag}_decode_input_tokens"]  # [5, 2]
        dec_expected = data[f"a14_{tag}_decode_logits"]  # [5, 2, 1, 64]

        for step in range(5):
            tok = jnp.asarray(dec_inputs[step][:, None], dtype=jnp.int32)
            dec_logits, cache, valid_mask = jitted_decode(
                tok, cur_pos, cache, valid_mask, jnp.asarray(prompt_len + step, dtype=jnp.int32)
            )
            assert dec_logits.shape == (2, 1, cfg.vocab_size)
            np.testing.assert_allclose(np.asarray(dec_logits), dec_expected[step], rtol=1e-4, atol=1e-4)
            cur_pos = cur_pos + 1

        assert trace_count == 1, f"decode_step recompiled across steps! trace_count={trace_count}"


def test_assignment_1_5(mod: ModuleType, data: np.lib.npyio.NpzFile) -> None:
    # 1. sample_next_token against golden greedy & top-5 sets
    logits = jnp.asarray(data["a15_sample_logits"])
    greedy = mod.sample_next_token(logits, jax.random.PRNGKey(0), temperature=0.0)
    np.testing.assert_array_equal(np.asarray(greedy), data["a15_sample_greedy_expected"])

    s1 = mod.sample_next_token(logits, jax.random.PRNGKey(7), temperature=0.8, top_k=5)
    s2 = mod.sample_next_token(logits, jax.random.PRNGKey(7), temperature=0.8, top_k=5)
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))

    top5_idx = data["a15_sample_top5_indices"]
    for b in range(logits.shape[0]):
        assert int(s1[b]) in set(top5_idx[b].tolist()), "top_k sampled outside top-k!"

    # 2. Greedy batched variable-length generate against golden output
    cfg = mod.ModelConfig.from_json(CONFIGS_DIR / "model_test.json", tie_word_embeddings=True)
    params = load_model_from_fixture(data, "m_tied", cfg.n_layers, True)
    prompt_ids = jnp.asarray(data["a15_prompt_ids"])
    prompt_mask = jnp.asarray(data["a15_prompt_mask"])
    expected_gen = data["a15_gen_expected"]
    max_new = expected_gen.shape[1]

    gen = mod.generate(params, prompt_ids, prompt_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
    assert gen.shape == expected_gen.shape, f"Expected shape {expected_gen.shape}, got {gen.shape}"
    np.testing.assert_array_equal(np.asarray(gen), expected_gen)

    # 3. Variable-length batched invariance: batched left-padded == unpadded B=1
    for i in range(prompt_ids.shape[0]):
        real_tokens = prompt_ids[i][prompt_mask[i]]
        single_ids = real_tokens[None, :]
        single_mask = jnp.ones_like(single_ids, dtype=jnp.bool_)
        single_gen = mod.generate(params, single_ids, single_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
        np.testing.assert_array_equal(
            np.asarray(gen[i]),
            np.asarray(single_gen[0]),
            err_msg=f"Variable-length batch item {i} diverged from unpadded B=1 generation!",
        )

    # 4. Edge case: max_new_tokens == 1
    gen_one = mod.generate(params, prompt_ids, prompt_mask, cfg, 1, jax.random.PRNGKey(0), temperature=0.0)
    assert gen_one.shape == (3, 1)
    np.testing.assert_array_equal(np.asarray(gen_one[:, 0]), expected_gen[:, 0])

    # 5. Stochastic determinism & valid vocabulary bounds
    gen_s1 = mod.generate(params, prompt_ids, prompt_mask, cfg, 8, jax.random.PRNGKey(42), temperature=0.9, top_k=10)
    gen_s2 = mod.generate(params, prompt_ids, prompt_mask, cfg, 8, jax.random.PRNGKey(42), temperature=0.9, top_k=10)
    np.testing.assert_array_equal(np.asarray(gen_s1), np.asarray(gen_s2))
    assert jnp.all((gen_s1 >= 0) & (gen_s1 < cfg.vocab_size))


def test_performance_kv_cache_speedup(mod: ModuleType, _data: np.lib.npyio.NpzFile) -> dict[str, float]:
    """Hard performance test (TPU v5e environment): KV-cached JIT+scan generate must beat uncached prefill loop."""
    cfg = mod.ModelConfig.from_json(CONFIGS_DIR / "model_tiny.json")
    params = mod.init_model_params(cfg, jax.random.PRNGKey(100), dtype=jnp.float32)

    bsz, prompt_len, max_new = 2, 16, 12
    prompt_ids = jax.random.randint(jax.random.PRNGKey(101), (bsz, prompt_len), 1, cfg.vocab_size, dtype=jnp.int32)
    prompt_mask = jnp.ones((bsz, prompt_len), dtype=jnp.bool_)
    cos, sin = mod.precompute_rope_freqs(prompt_len + max_new, cfg.head_dim, cfg.rope_theta)

    # Warm up student's JIT-compiled generate once (exclude compile time from steady-state latency)
    warm = mod.generate(params, prompt_ids, prompt_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
    warm.block_until_ready()

    t0 = time.perf_counter()
    cached_out = mod.generate(params, prompt_ids, prompt_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
    cached_out.block_until_ready()
    t_cached = time.perf_counter() - t0

    # Uncached O(T^2) baseline using student's own prefill at every step (no reference code leaked)
    t0 = time.perf_counter()
    cur_ids = prompt_ids
    cur_mask = prompt_mask
    uncached_tokens = []
    for _ in range(max_new):
        logits, _, _, _ = mod.prefill(params, cur_ids, cur_mask, cfg, cur_ids.shape[1], cos, sin)
        next_tok = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)
        uncached_tokens.append(next_tok)
        cur_ids = jnp.concatenate([cur_ids, next_tok[:, None]], axis=1)
        cur_mask = jnp.concatenate([cur_mask, jnp.ones((bsz, 1), dtype=jnp.bool_)], axis=1)
    uncached_out = jnp.stack(uncached_tokens, axis=1)
    uncached_out.block_until_ready()
    t_uncached = time.perf_counter() - t0

    np.testing.assert_array_equal(np.asarray(cached_out), np.asarray(uncached_out))

    speedup = t_uncached / max(t_cached, 1e-9)
    assert speedup >= 100.0, (
        f"Expected KV-cached JIT+scan generate to be >= 100.0x faster than uncached recomputation on TPU v5e, "
        f"but got {speedup:.2f}x (cached={t_cached*1e3:.1f} ms, uncached={t_uncached*1e3:.1f} ms)."
    )
    return {
        "t_cached_ms": t_cached * 1e3,
        "t_uncached_ms": t_uncached * 1e3,
        "speedup": speedup,
    }


def run_roofline_benchmark(mod: ModuleType) -> None:
    """Optional TPU v5e / CPU batch-size roofline benchmark (`--benchmark`)."""
    print("\n=== Batch-Size Roofline Benchmark (TPU v5e Environment) ===")
    device = jax.devices()[0]
    dtype = jnp.bfloat16 if device.platform in ("tpu", "gpu") else jnp.float32
    dtype_bytes = 2 if dtype == jnp.bfloat16 else 4

    cfg = mod.ModelConfig.from_json(CONFIGS_DIR / "model_benchmark.json")
    params = mod.init_model_params(cfg, jax.random.PRNGKey(2026), dtype=dtype)
    prompt_len = 64
    max_new = 64

    print(f"Device: {device} | dtype: {dtype} | prompt_len={prompt_len}, max_new={max_new}")
    print(f"{'Batch':>6} | {'Total (ms)':>10} | {'TPOT (ms)':>10} | {'Tok/s':>10} | {'Roofline TPOT (ms)':>18}")
    print("-" * 68)
    for bsz in (1, 4, 16, 32):
        prompt_ids = jax.random.randint(jax.random.PRNGKey(bsz), (bsz, prompt_len), 1, cfg.vocab_size, dtype=jnp.int32)
        prompt_mask = jnp.ones((bsz, prompt_len), dtype=jnp.bool_)

        out = mod.generate(params, prompt_ids, prompt_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
        out.block_until_ready()

        t0 = time.perf_counter()
        out = mod.generate(params, prompt_ids, prompt_mask, cfg, max_new, jax.random.PRNGKey(0), temperature=0.0)
        out.block_until_ready()
        elapsed_s = time.perf_counter() - t0

        tpot_ms = (elapsed_s / max_new) * 1e3
        tok_per_s = (bsz * max_new) / elapsed_s
        roof = mod.estimate_tpu_v5e_roofline(cfg, bsz, prompt_len + max_new // 2, dtype_bytes=dtype_bytes)
        print(
            f"{bsz:>6} | {elapsed_s*1e3:>10.2f} | {tpot_ms:>10.3f} | {tok_per_s:>10.1f} | {roof['roofline_tpot_ms']:>18.4f}"
        )


SECTION_TESTS = {
    "1.1": ("Assignment 1.1 (RMSNorm, RoPE, SwiGLU)", test_assignment_1_1, True),
    "1.2": ("Assignment 1.2 (Static KV Cache & dynamic_update_slice)", test_assignment_1_2, True),
    "1.3": ("Assignment 1.3 (Masking & Grouped-Query Attention)", test_assignment_1_3, True),
    "1.4": ("Assignment 1.4 (Transformer Block, Prefill, Decode Step)", test_assignment_1_4, True),
    "1.5": ("Assignment 1.5 (Sampling, Batched Variable-Length Generate)", test_assignment_1_5, True),
    "perf": ("Performance    (TPU v5e KV-Cache JIT+scan >= 100x Speedup)", test_performance_kv_cache_speedup, False),
}


def _resolve_submission(submission: str | Path | ModuleType) -> tuple[ModuleType, str]:
    if isinstance(submission, ModuleType):
        return submission, getattr(submission, "__file__", submission.__name__)
    sub_label = str(submission)
    return load_submission(sub_label), sub_label


def run_section(
    submission: str | Path | ModuleType = DEFAULT_MODEL_PATH,
    section: str = "1.1",
    data_path: str | Path | None = None,
) -> int:
    """Run a single assignment section test (`"1.1"` .. `"1.5"` or `"perf"`) in-process."""
    if section not in SECTION_TESTS:
        raise ValueError(f"Unknown section {section!r}. Choose from {list(SECTION_TESTS)}")

    resolved_data = Path(data_path).resolve() if data_path is not None else DEFAULT_DATA_PATH
    if not resolved_data.exists():
        raise FileNotFoundError(f"Golden test data not found: {resolved_data}")

    mod, _ = _resolve_submission(submission)
    name, fn, high_precision = SECTION_TESTS[section]

    with np.load(resolved_data) as data:
        try:
            if high_precision:
                with jax.default_matmul_precision("highest"):
                    result = fn(mod, data)
            else:
                result = fn(mod, data)
            extra = ""
            if isinstance(result, dict) and "speedup" in result:
                extra = (
                    f" [cached={result['t_cached_ms']:.1f} ms, "
                    f"uncached={result['t_uncached_ms']:.1f} ms, "
                    f"speedup={result['speedup']:.1f}x]"
                )
            print(f"[PASS] {name}{extra}")
            return 0
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            raise


def run_all(
    submission: str | Path | ModuleType = DEFAULT_MODEL_PATH,
    data_path: str | Path | None = None,
    benchmark: bool = False,
) -> int:
    """Run all correctness and performance tests in-process.

    Accepts a path to the `model/` package directory, a single `.py` module,
    or an already-imported module object (for in-process TPU notebook runs).
    """
    resolved_data = Path(data_path).resolve() if data_path is not None else DEFAULT_DATA_PATH
    if not resolved_data.exists():
        raise FileNotFoundError(f"Golden test data not found: {resolved_data}")

    mod, sub_label = _resolve_submission(submission)

    print(f"Loading submission: {sub_label}")
    print(f"Loading test data:  {resolved_data}")
    print(f"JAX {jax.__version__} | Devices: {jax.devices()}\n")

    tests = list(SECTION_TESTS.values())

    passed = 0
    with np.load(resolved_data) as data:
        for name, fn, high_precision in tests:
            try:
                if high_precision:
                    # On TPUs, float32 matmul defaults to bfloat16 MXU precision;
                    # use IEEE float32 ("highest") for tight golden-data checks.
                    with jax.default_matmul_precision("highest"):
                        result = fn(mod, data)
                else:
                    result = fn(mod, data)
                extra = ""
                if isinstance(result, dict) and "speedup" in result:
                    extra = (
                        f" [cached={result['t_cached_ms']:.1f} ms, "
                        f"uncached={result['t_uncached_ms']:.1f} ms, "
                        f"speedup={result['speedup']:.1f}x]"
                    )
                print(f"[PASS] {name}{extra}")
                passed += 1
            except Exception as exc:
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    print(f"\nSummary: {passed}/{len(tests)} tests passed.")
    if benchmark and passed == len(tests):
        run_roofline_benchmark(mod)
    return 0 if passed == len(tests) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Student-safe test runner for CMU 11-868 LLM Inference HW.")
    parser.add_argument(
        "submission",
        nargs="?",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to student `model/` package directory or `.py` file (default: model/)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to golden test fixtures (default: data/test_data.npz)",
    )
    parser.add_argument("--benchmark", action="store_true", help="Also run batch-size roofline benchmark")
    args = parser.parse_args(argv)
    return run_all(args.submission, data_path=args.data, benchmark=args.benchmark)


if __name__ == "__main__":
    raise SystemExit(main())

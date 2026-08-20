# SPDX-License-Identifier: MIT
"""Tests for the node layer: the TQ3 cache wrapper and the attention patch.

These paths had no coverage, and they are the ones ComfyUI actually calls.
"""

import pytest
import torch

import tq3_core
import turboquant_nodes


BLOCK = tq3_core.TQ3_BLOCK


def _cosine(x, xr):
    return (
        torch.nn.functional.cosine_similarity(
            x.reshape(-1, x.shape[-1]).float(),
            xr.reshape(-1, xr.shape[-1]).float(),
            dim=-1,
        )
        .mean()
        .item()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_wrapper_roundtrip_preserves_dtype_and_shape(dtype):
    wrapper = turboquant_nodes.TQ3KVCacheWrapper()
    tensor = torch.randn(2, 8, BLOCK, dtype=dtype)

    wrapper.store("k", tensor)
    loaded = wrapper.load("k")

    assert loaded.shape == tensor.shape
    assert loaded.dtype == dtype
    assert _cosine(tensor, loaded) > 0.97


@pytest.mark.parametrize("dim", [320, 640, 1280, BLOCK + 1])
def test_wrapper_pads_dims_that_are_not_a_multiple_of_the_block(dim):
    """SD/SDXL inner dims are not multiples of 128, so this is the common path."""
    wrapper = turboquant_nodes.TQ3KVCacheWrapper()
    tensor = torch.randn(2, 4, dim)

    wrapper.store("k", tensor)
    loaded = wrapper.load("k")

    assert loaded.shape == tensor.shape
    assert _cosine(tensor, loaded) > 0.97


def test_wrapper_roundtrip_keeps_structured_tensors():
    """A per-channel offset is exactly what real attention tensors carry."""
    wrapper = turboquant_nodes.TQ3KVCacheWrapper()
    tensor = torch.randn(2, 8, BLOCK) + torch.linspace(-4.0, 4.0, BLOCK)

    wrapper.store("k", tensor)

    assert _cosine(tensor, wrapper.load("k")) > 0.97


def test_wrapper_stores_raw_tensors_when_disabled():
    wrapper = turboquant_nodes.TQ3KVCacheWrapper(enabled=False)
    tensor = torch.randn(2, BLOCK)

    wrapper.store("k", tensor)

    assert wrapper.load("k") is tensor
    assert wrapper.compression_ratio == 0.0


def test_wrapper_load_returns_none_for_an_unknown_key():
    assert turboquant_nodes.TQ3KVCacheWrapper().load("missing") is None


def test_wrapper_reports_the_documented_ratio_for_fp16():
    wrapper = turboquant_nodes.TQ3KVCacheWrapper()

    wrapper.store("k", torch.randn(4, 8, BLOCK, dtype=torch.float16))

    # 128 fp16 values (256 bytes) -> 48 + 4 + 4 = 56 bytes
    assert wrapper.compression_ratio == pytest.approx(256 / 56, rel=1e-6)
    assert wrapper.savings_mb > 0
    assert "Ratio:" in wrapper.stats_string()


def test_wrapper_clear_resets_cache_and_stats():
    wrapper = turboquant_nodes.TQ3KVCacheWrapper()
    wrapper.store("k", torch.randn(2, BLOCK))

    wrapper.clear()

    assert wrapper.load("k") is None
    assert wrapper.compression_ratio == 0.0


def test_attn_patch_passes_through_a_none_context():
    """ComfyUI's attn2 path does not coerce a None context (attn1 does)."""
    wrapper = turboquant_nodes.TQ3KVCacheWrapper()
    patch = turboquant_nodes._make_attn_patch(wrapper)
    hidden = torch.randn(2, 16, 320)

    q, k, v = patch(hidden, None, None, {})

    assert q is hidden
    assert k is None
    assert v is None


def test_attn_patch_roundtrips_kv_and_leaves_no_cache_entries():
    wrapper = turboquant_nodes.TQ3KVCacheWrapper()
    patch = turboquant_nodes._make_attn_patch(wrapper)
    hidden = torch.randn(2, 16, 320)

    q, k, v = patch(hidden, hidden, hidden, {})

    assert q is hidden
    assert k.shape == hidden.shape and v.shape == hidden.shape
    assert _cosine(hidden, k) > 0.97
    assert wrapper._cache == {}
    assert wrapper._stats["num_stores"] == 2


def test_attn_patch_skips_tensors_narrower_than_one_block():
    wrapper = turboquant_nodes.TQ3KVCacheWrapper()
    patch = turboquant_nodes._make_attn_patch(wrapper)
    narrow = torch.randn(2, 16, 64)

    q, k, v = patch(narrow, narrow, narrow, {})

    assert k is narrow and v is narrow
    assert wrapper._stats["num_stores"] == 0


def test_node_registration_exposes_both_documented_nodes():
    assert set(turboquant_nodes.NODE_CLASS_MAPPINGS) == {
        "TurboQuantPatch",
        "TurboQuantInfo",
    }
    assert set(turboquant_nodes.NODE_DISPLAY_NAME_MAPPINGS) == {
        "TurboQuantPatch",
        "TurboQuantInfo",
    }
    for name, cls in turboquant_nodes.NODE_CLASS_MAPPINGS.items():
        assert hasattr(cls, cls.FUNCTION), name
        assert "required" in cls.INPUT_TYPES()


def test_info_node_reports_stats_once_a_wrapper_is_active():
    turboquant_nodes._active_wrapper = None
    try:
        (idle,) = turboquant_nodes.TurboQuantInfo().info(model=None)
        assert "Not yet active" in idle

        wrapper = turboquant_nodes.TQ3KVCacheWrapper()
        wrapper.store("k", torch.randn(2, BLOCK, dtype=torch.float16))
        turboquant_nodes._active_wrapper = wrapper

        (active,) = turboquant_nodes.TurboQuantInfo().info(model=None)
        assert "Stores:      1" in active
    finally:
        turboquant_nodes._active_wrapper = None

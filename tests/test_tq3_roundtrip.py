# SPDX-License-Identifier: MIT
"""Round-trip quality tests for TQ3.

The pre-existing suite covers payload shapes and argument validation only, so a
reconstruction that is worse than returning zeros still passed. These tests pin
the property the README advertises -- cosine similarity > 0.97 -- across blocks
that are *not* white noise, and check the transform against independent
references rather than against itself.
"""

import math

import pytest
import torch

import tq3_core


QUALITY_TARGET = 0.97  # README: "Round-trip cosine similarity: >0.97"


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


def _hadamard_matrix(n):
    """Normalized Hadamard matrix built by Sylvester recursion.

    Independent of _fwht_inplace, so it can be used as a reference for it.
    """
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return h / math.sqrt(n)


def _structured_blocks():
    """Block shapes that carry real structure, not just Gaussian noise."""
    torch.manual_seed(7)
    block = tq3_core.TQ3_BLOCK
    ramp = torch.linspace(0.0, 2 * math.pi, block)

    outlier = torch.randn(16, block)
    outlier[:, 7] = 40.0

    channelwise = torch.randn(16, block) * torch.linspace(0.2, 4.0, block)
    channelwise += torch.linspace(-2.0, 2.0, block)

    return {
        "gaussian": torch.randn(16, block),
        "constant": torch.full((16, block), 2.5),
        "dc_offset": torch.randn(16, block) + 3.0,
        "negative_dc_offset": torch.randn(16, block) - 5.0,
        "low_frequency": torch.stack([torch.sin(ramp * (1 + i * 0.1)) for i in range(16)]),
        "linear_ramp": torch.stack([ramp * (1 + i) for i in range(16)]),
        "outlier_channel": outlier,
        "per_channel_scale_and_offset": channelwise,
        "one_hot": torch.eye(block)[:16],
        "tiny_magnitudes": torch.randn(16, block) * 1e-4,
        "large_magnitudes": torch.randn(16, block) * 1e4,
    }


@pytest.mark.parametrize("name", sorted(_structured_blocks()))
def test_roundtrip_meets_quality_target_for_structured_blocks(name):
    x = _structured_blocks()[name]

    xr = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(x))

    cosine = _cosine(x, xr)
    assert cosine > QUALITY_TARGET, f"{name}: cosine {cosine:.4f}"


@pytest.mark.parametrize("name", sorted(_structured_blocks()))
def test_roundtrip_error_stays_below_the_signal_for_structured_blocks(name):
    """A relative L2 error >= 1.0 means returning zeros would be no worse."""
    x = _structured_blocks()[name]

    xr = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(x))

    relative_error = ((x - xr).norm() / x.norm()).item()
    assert relative_error < 0.35, f"{name}: relative L2 error {relative_error:.4f}"


def _roundtrip_without_sign_flips(x):
    original = tq3_core._generate_sign_flips
    tq3_core._generate_sign_flips = lambda dim, seed=42, device="cpu": torch.ones(
        dim, device=device
    )
    try:
        return tq3_core.tq3_dequantize(tq3_core.tq3_quantize(x))
    finally:
        tq3_core._generate_sign_flips = original


def test_sign_flips_change_the_reconstruction():
    """The sign flips must be applied before the FWHT, where they do work.

    Applied afterwards they are a no-op: absmax is invariant under a sign flip
    and the codebook is symmetric about zero, so the reconstruction comes back
    bit-for-bit identical while low-frequency blocks stay collapsed. Any input
    without exact zeros in the transform output shows that, because a value of
    exactly 0.0 is the one place a flip can change the quantized index.
    """
    torch.manual_seed(17)
    x = torch.randn(8, tq3_core.TQ3_BLOCK) + 3.0

    with_flips = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(x))
    without_flips = _roundtrip_without_sign_flips(x)

    assert not torch.allclose(with_flips, without_flips), (
        "disabling the sign flips left the reconstruction unchanged, so they "
        "are not being applied before the transform"
    )


def test_sign_flips_are_what_rescues_a_low_frequency_block():
    x = torch.full((4, tq3_core.TQ3_BLOCK), 2.5)

    with_flips = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(x))
    without_flips = _roundtrip_without_sign_flips(x)

    assert _cosine(x, with_flips) > QUALITY_TARGET
    # Without the randomization the whole block collapses into one coefficient
    # and the reconstruction is worse than returning zeros.
    assert _cosine(x, without_flips) < 0.7
    assert ((x - without_flips).norm() / x.norm()).item() > 1.0


def test_sign_flips_are_deterministic_and_are_signs():
    first = tq3_core._generate_sign_flips(tq3_core.TQ3_BLOCK)
    second = tq3_core._generate_sign_flips(tq3_core.TQ3_BLOCK)

    assert torch.equal(first, second)
    assert torch.equal(first.abs(), torch.ones_like(first))
    assert tq3_core._generate_sign_flips(tq3_core.TQ3_BLOCK, seed=43) is not None
    assert not torch.equal(
        first, tq3_core._generate_sign_flips(tq3_core.TQ3_BLOCK, seed=43)
    )


def test_fwht_matches_an_independent_hadamard_matrix():
    torch.manual_seed(11)
    x = torch.randn(5, tq3_core.TQ3_BLOCK)
    reference = x @ _hadamard_matrix(tq3_core.TQ3_BLOCK).T

    transformed = tq3_core._fwht_inplace(x.clone())

    assert torch.allclose(transformed, reference, atol=1e-5)


def test_quantize_to_indices_picks_the_nearest_centroid():
    values = torch.linspace(-1.0, 1.0, steps=4097)

    indices = tq3_core._quantize_to_indices(values, tq3_core.TQ3_BOUNDARIES)
    nearest = (values.unsqueeze(-1) - tq3_core.TQ3_CENTROIDS).abs().argmin(dim=-1)

    assert torch.equal(indices.long(), nearest)


def test_quantize_is_deterministic():
    torch.manual_seed(3)
    x = torch.randn(4, 256)

    first = tq3_core.tq3_quantize(x)
    second = tq3_core.tq3_quantize(x)

    assert torch.equal(first["packed"], second["packed"])
    assert torch.equal(first["scales"], second["scales"])
    assert torch.equal(first["norms"], second["norms"])


@pytest.mark.parametrize("shape", [(128,), (3, 128), (2, 3, 384), (1, 1, 1, 512)])
def test_roundtrip_preserves_shape(shape):
    x = torch.randn(*shape)

    xr = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(x))

    assert xr.shape == x.shape
    assert xr.dtype == torch.float32


def test_roundtrip_accepts_non_contiguous_input():
    x = torch.randn(4, tq3_core.TQ3_BLOCK, 8).transpose(1, 2)
    assert not x.is_contiguous()

    xr = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(x))

    assert xr.shape == x.shape
    assert _cosine(x, xr) > QUALITY_TARGET


def test_roundtrip_handles_an_all_zero_block():
    x = torch.zeros(2, tq3_core.TQ3_BLOCK)

    xr = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(x))

    assert torch.allclose(xr, torch.zeros_like(xr), atol=1e-6)


def test_blocks_are_quantized_independently():
    """Concatenating blocks must not change either block's reconstruction."""
    torch.manual_seed(5)
    a = torch.randn(1, tq3_core.TQ3_BLOCK)
    b = torch.randn(1, tq3_core.TQ3_BLOCK) * 1000.0

    together = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(torch.cat([a, b], dim=-1)))
    alone = tq3_core.tq3_dequantize(tq3_core.tq3_quantize(a))

    assert torch.allclose(together[:, : tq3_core.TQ3_BLOCK], alone, atol=1e-4)


def test_memory_bytes_matches_the_documented_block_budget():
    """48 bytes of indices + 4 byte norm + 4 byte scale per 128 values."""
    block = tq3_core.TQ3_BLOCK
    compressed, original, ratio = tq3_core.tq3_memory_bytes((7, 4 * block))

    assert compressed == 7 * 4 * 56
    assert original == 7 * 4 * block * 2  # FP16 baseline
    assert ratio == pytest.approx(2 * block / 56)

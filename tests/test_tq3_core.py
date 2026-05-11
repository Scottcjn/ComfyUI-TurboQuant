import re

import pytest
import torch

import tq3_core


def test_pack_unpack_roundtrip_preserves_batched_indices():
    indices = torch.arange(tq3_core.TQ3_BLOCK * 3, dtype=torch.uint8).reshape(
        3, tq3_core.TQ3_BLOCK
    )
    indices = indices % 8

    packed = tq3_core._pack_3bit(indices)
    unpacked = tq3_core._unpack_3bit(packed)

    assert packed.shape == (3, 48)
    assert packed.dtype == torch.uint8
    assert torch.equal(unpacked, indices)


def test_tq3_memory_bytes_accounts_for_vectors_and_blocks():
    compressed, original, ratio = tq3_core.tq3_memory_bytes((2, 3, 256))

    assert compressed == 672
    assert original == 3072
    assert ratio == pytest.approx(original / compressed)


@pytest.mark.parametrize("shape", [(), (0, 128), (2, 0), (2, 127), (2, -128)])
def test_tq3_memory_bytes_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        tq3_core.tq3_memory_bytes(shape)


@pytest.mark.parametrize(
    ("bad_input", "expected_error"),
    [
        ([1.0] * tq3_core.TQ3_BLOCK, TypeError),
        (torch.tensor(1.0), ValueError),
        (torch.ones(2, 127), ValueError),
    ],
)
def test_tq3_quantize_rejects_invalid_inputs(bad_input, expected_error):
    with pytest.raises(expected_error):
        tq3_core.tq3_quantize(bad_input)


def test_tq3_quantize_reports_expected_payload_shapes_for_two_blocks():
    x = torch.linspace(-1.0, 1.0, steps=512).reshape(2, 256)

    payload = tq3_core.tq3_quantize(x)

    assert payload["norms"].shape == (2, 2)
    assert payload["scales"].shape == (2, 2)
    assert payload["packed"].shape == (2, 2, 48)
    assert payload["orig_shape"] == x.shape


def test_tq3_dequantize_rejects_missing_required_payload_key():
    with pytest.raises(KeyError, match="scales"):
        tq3_core.tq3_dequantize(
            {
                "norms": torch.ones(1),
                "packed": torch.zeros(1, 48, dtype=torch.uint8),
                "orig_shape": (128,),
            }
        )


def test_tq3_dequantize_rejects_mismatched_payload_shapes():
    payload = {
        "norms": torch.ones(2),
        "scales": torch.ones(1),
        "packed": torch.zeros(2, 48, dtype=torch.uint8),
        "orig_shape": (256,),
    }

    with pytest.raises(ValueError, match="identical shapes"):
        tq3_core.tq3_dequantize(payload)

    payload["scales"] = torch.ones(2)
    payload["packed"] = torch.zeros(1, 48, dtype=torch.uint8)

    with pytest.raises(ValueError, match=re.escape("norms/scales shape + [48]")):
        tq3_core.tq3_dequantize(payload)

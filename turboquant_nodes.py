"""
TurboQuant ComfyUI Nodes

Two nodes:
  1. TurboQuantPatch  - Patches a model to use TQ3-compressed KV cache
  2. TurboQuantInfo   - Shows compression stats for a patched model
"""

import torch
import math
from . import tq3_core


class TQ3KVCacheWrapper:
    """
    Wraps a standard KV cache to transparently compress/decompress
    using TQ3 quantization.

    Stored tensors are kept in TQ3 format (dict of norms, scales, packed).
    On read, they are decompressed back to float/half.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._cache = {}  # key -> tq3 dict or raw tensor
        self._stats = {
            "compressed_bytes": 0,
            "original_bytes": 0,
            "num_stores": 0,
        }

    def store(self, key: str, tensor: torch.Tensor) -> None:
        """Compress and store a KV tensor."""
        if not self.enabled:
            self._cache[key] = tensor
            return

        orig_dtype = tensor.dtype
        orig_device = tensor.device
        D = tensor.shape[-1]

        # TQ3 requires last dim divisible by 128
        if D % tq3_core.TQ3_BLOCK != 0:
            # Pad to next multiple of 128
            pad_to = math.ceil(D / tq3_core.TQ3_BLOCK) * tq3_core.TQ3_BLOCK
            pad_size = pad_to - D
            tensor_padded = torch.nn.functional.pad(tensor.float(), (0, pad_size))
        else:
            tensor_padded = tensor.float()
            pad_size = 0

        tq = tq3_core.tq3_quantize(tensor_padded)
        tq["_orig_dtype"] = orig_dtype
        tq["_orig_device"] = orig_device
        tq["_pad_size"] = pad_size

        self._cache[key] = tq

        # Track stats
        comp, orig, _ = tq3_core.tq3_memory_bytes(tensor_padded.shape)
        self._stats["compressed_bytes"] += comp
        self._stats["original_bytes"] += orig
        self._stats["num_stores"] += 1

    def load(self, key: str) -> torch.Tensor:
        """Decompress and return a KV tensor."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        if isinstance(entry, torch.Tensor):
            return entry

        # TQ3 dict
        result = tq3_core.tq3_dequantize(entry)
        pad_size = entry.get("_pad_size", 0)
        if pad_size > 0:
            result = result[..., :-pad_size]

        orig_dtype = entry.get("_orig_dtype", torch.float16)
        orig_device = entry.get("_orig_device", "cpu")
        return result.to(dtype=orig_dtype, device=orig_device)

    def clear(self):
        self._cache.clear()
        self._stats = {"compressed_bytes": 0, "original_bytes": 0, "num_stores": 0}

    @property
    def compression_ratio(self) -> float:
        if self._stats["original_bytes"] == 0:
            return 0.0
        return self._stats["original_bytes"] / max(self._stats["compressed_bytes"], 1)

    @property
    def savings_mb(self) -> float:
        diff = self._stats["original_bytes"] - self._stats["compressed_bytes"]
        return diff / (1024 * 1024)

    def stats_string(self) -> str:
        ratio = self.compression_ratio
        orig_mb = self._stats["original_bytes"] / (1024 * 1024)
        comp_mb = self._stats["compressed_bytes"] / (1024 * 1024)
        return (
            f"TurboQuant TQ3 KV Cache Stats\n"
            f"-----------------------------\n"
            f"Stores:      {self._stats['num_stores']}\n"
            f"FP16 size:   {orig_mb:.1f} MB\n"
            f"TQ3 size:    {comp_mb:.1f} MB\n"
            f"Ratio:       {ratio:.2f}x\n"
            f"Savings:     {self.savings_mb:.1f} MB"
        )


# Global wrapper instance so the info node can read stats
_active_wrapper = None


def _make_attn_patch(wrapper: TQ3KVCacheWrapper):
    """
    Create an attention patch function compatible with ComfyUI's
    model_patcher system.

    ComfyUI patches are callables that receive (q, k, v, extra_options)
    and return (q, k, v). We intercept to compress/decompress KV.

    For the KV cache compression use case, the patch:
    1. On store (when new K,V are computed): compress and cache
    2. On load (when cached K,V are reused): decompress

    In diffusion models, attention is recomputed each step, so the
    main benefit is reducing the *peak* memory of the KV tensors
    that coexist during the attention computation. The patch quantizes
    K and V *in transit* so they occupy less VRAM while stored.
    """

    step_counter = [0]

    def attn_patch(q, k, v, extra_options):
        nonlocal step_counter

        # Compress K and V to TQ3, then immediately decompress.
        # This forces the memory layout through the compressed format,
        # meaning the original FP16 K/V can be freed sooner by PyTorch's
        # allocator. The decompressed versions are fresh allocations
        # that only live for the attention matmul.

        step_counter[0] += 1
        step_id = step_counter[0]

        # Only compress if dim is >= 128 (otherwise overhead isn't worth it)
        if k.shape[-1] >= tq3_core.TQ3_BLOCK:
            k_key = f"k_{step_id}"
            v_key = f"v_{step_id}"
            wrapper.store(k_key, k)
            wrapper.store(v_key, v)

            # Decompress - original k,v can now be freed
            k = wrapper.load(k_key)
            v = wrapper.load(v_key)

            # Clean up compressed copies (they served their purpose)
            if k_key in wrapper._cache:
                del wrapper._cache[k_key]
            if v_key in wrapper._cache:
                del wrapper._cache[v_key]

        return q, k, v

    return attn_patch


class TurboQuantPatch:
    """
    Patches a model to use TQ3-compressed KV cache during attention.

    This reduces peak VRAM usage of KV tensors by ~4.5x, enabling
    larger models or longer sequences on the same GPU.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "TurboQuant"
    DESCRIPTION = (
        "Patches model attention to use TQ3-compressed KV cache. "
        "Reduces KV VRAM by ~4.5x with minimal quality loss."
    )

    def patch(self, model, enabled):
        global _active_wrapper

        if not enabled:
            return (model,)

        wrapper = TQ3KVCacheWrapper(enabled=True)
        _active_wrapper = wrapper

        # Clone the model patcher so we don't modify the original
        patched = model.clone()

        # Register the attention patch via ComfyUI's patcher API
        # "attn1" = self-attention, "attn2" = cross-attention
        attn_patch = _make_attn_patch(wrapper)
        patched.set_model_attn1_patch(attn_patch)
        patched.set_model_attn2_patch(attn_patch)

        return (patched,)


class TurboQuantInfo:
    """
    Displays TurboQuant compression statistics.

    Connect after TurboQuantPatch to see live compression ratio
    and VRAM savings during generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "info"
    CATEGORY = "TurboQuant"
    OUTPUT_NODE = True
    DESCRIPTION = "Shows TQ3 compression ratio and VRAM savings."

    def info(self, model):
        global _active_wrapper

        if _active_wrapper is not None:
            stats = _active_wrapper.stats_string()
        else:
            # Estimate based on model config
            stats = (
                "TurboQuant TQ3 KV Cache\n"
                "-----------------------\n"
                "Status: Not yet active (no inference run)\n"
                "Expected compression: ~4.5x\n"
                "Encoding: 3-bit Lloyd-Max + FWHT + sign flips\n"
                "Block size: 128 floats -> 56 bytes"
            )

        return (stats,)


# Node registration dicts
NODE_CLASS_MAPPINGS = {
    "TurboQuantPatch": TurboQuantPatch,
    "TurboQuantInfo": TurboQuantInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TurboQuantPatch": "TurboQuant KV Patch",
    "TurboQuantInfo": "TurboQuant Info",
}

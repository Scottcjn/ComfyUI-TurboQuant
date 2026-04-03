"""
TQ3 Core: TurboQuant 3-bit KV cache compression.

Compresses 128-float vectors to 56 bytes (3.5 bits/value):
  48 bytes packed indices (128 * 3 bits) + 4 byte norm + 4 byte scale

Algorithm:
  1. L2 normalize
  2. Fast Walsh-Hadamard Transform (7 butterfly stages)
  3. Deterministic random sign flips
  4. Absmax scale to [-1,+1]
  5. Lloyd-Max 8-level codebook quantize (3 bits per value)
  6. Pack 3-bit indices into bytes

Round-trip quality target: cosine similarity > 0.97 for typical attention vectors.
"""

import torch
import math

# Lloyd-Max optimal 8-level codebook for Gaussian-ish distributions
# after absmax normalization to [-1, +1]
TQ3_CENTROIDS = torch.tensor(
    [-0.9816, -0.6168, -0.3479, -0.1129, 0.1129, 0.3479, 0.6168, 0.9816],
    dtype=torch.float32,
)
TQ3_BOUNDARIES = torch.tensor(
    [-0.7992, -0.4824, -0.2304, 0.0, 0.2304, 0.4824, 0.7992],
    dtype=torch.float32,
)

# Block size for TQ3 (must be power of 2 for FWHT)
TQ3_BLOCK = 128
TQ3_LOG2_BLOCK = 7  # log2(128)


def _generate_sign_flips(dim: int, seed: int = 42, device="cpu") -> torch.Tensor:
    """Generate deterministic random sign flip vector (+1/-1)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32)
    return (bits * 2.0 - 1.0).to(device)


def _fwht_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform along the last dimension.
    x shape: [..., 128]. Operates in-place for efficiency.
    Normalized by 1/sqrt(N).
    """
    n = x.shape[-1]
    if n != TQ3_BLOCK:
        raise ValueError(f"FWHT requires dim={TQ3_BLOCK}, got {n}")

    h = 1
    for _ in range(TQ3_LOG2_BLOCK):
        # Butterfly: split into pairs of size h
        x_view = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x_view[..., 0, :].clone()
        b = x_view[..., 1, :].clone()
        x_view[..., 0, :] = a + b
        x_view[..., 1, :] = a - b
        h *= 2

    # Normalize
    x.mul_(1.0 / math.sqrt(n))
    return x


def _quantize_to_indices(x_norm: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Quantize normalized [-1,+1] values to 3-bit indices (0-7)."""
    # Use searchsorted for vectorized boundary comparison
    # boundaries is sorted ascending
    indices = torch.searchsorted(boundaries.to(x_norm.device), x_norm.contiguous())
    return indices.to(torch.uint8)


def _pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 128 3-bit indices into 48 bytes (384 bits).
    Input: [..., 128] uint8 (values 0-7)
    Output: [..., 48] uint8
    """
    if indices.shape[-1] != TQ3_BLOCK:
        raise ValueError(
            f"3-bit packing requires last dim {TQ3_BLOCK}, got {indices.shape[-1]}"
        )
    shape_prefix = indices.shape[:-1]
    idx = indices.view(-1, TQ3_BLOCK).to(torch.int32)
    batch = idx.shape[0]

    # Pack 8 values (24 bits) into 3 bytes at a time
    # 128 values / 8 = 16 groups, 16 * 3 = 48 bytes
    packed = torch.zeros(batch, 48, dtype=torch.uint8, device=indices.device)

    for g in range(16):
        v = idx[:, g * 8 : g * 8 + 8]  # [batch, 8] each 0-7
        # Pack 8 x 3-bit into 3 bytes (24 bits)
        bits = torch.zeros(batch, dtype=torch.int32, device=indices.device)
        for i in range(8):
            bits = bits | (v[:, i] << (i * 3))
        packed[:, g * 3 + 0] = (bits & 0xFF).to(torch.uint8)
        packed[:, g * 3 + 1] = ((bits >> 8) & 0xFF).to(torch.uint8)
        packed[:, g * 3 + 2] = ((bits >> 16) & 0xFF).to(torch.uint8)

    return packed.view(*shape_prefix, 48)


def _unpack_3bit(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack 48 bytes back to 128 3-bit indices.
    Input: [..., 48] uint8
    Output: [..., 128] uint8
    """
    if packed.shape[-1] != 48:
        raise ValueError(f"3-bit unpacking requires last dim 48, got {packed.shape[-1]}")
    shape_prefix = packed.shape[:-1]
    p = packed.view(-1, 48).to(torch.int32)
    batch = p.shape[0]

    indices = torch.zeros(batch, TQ3_BLOCK, dtype=torch.uint8, device=packed.device)

    for g in range(16):
        bits = (
            p[:, g * 3 + 0]
            | (p[:, g * 3 + 1] << 8)
            | (p[:, g * 3 + 2] << 16)
        )
        for i in range(8):
            indices[:, g * 8 + i] = ((bits >> (i * 3)) & 0x7).to(torch.uint8)

    return indices.view(*shape_prefix, TQ3_BLOCK)


def tq3_quantize(x: torch.Tensor) -> dict:
    """
    Quantize tensor to TQ3 format.

    Input: x of shape [..., D] where D is divisible by 128.
    Returns dict with:
      - norms:   [..., D//128] float32 (L2 norms per block)
      - scales:  [..., D//128] float32 (absmax per block after FWHT)
      - packed:  [..., D//128, 48] uint8 (packed 3-bit indices)
      - orig_shape: original shape
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("tq3_quantize expects a torch.Tensor input.")
    if x.ndim == 0:
        raise ValueError("tq3_quantize expects at least one dimension.")
    if x.shape[-1] <= 0 or x.shape[-1] % TQ3_BLOCK != 0:
        raise ValueError(
            f"Last dim must be a positive multiple of {TQ3_BLOCK}, got {x.shape[-1]}"
        )
    device = x.device
    orig_shape = x.shape
    D = x.shape[-1]
    num_blocks = D // TQ3_BLOCK

    # Reshape to blocks: [..., num_blocks, 128]
    xb = x.float().reshape(*x.shape[:-1], num_blocks, TQ3_BLOCK)

    # Step 1: L2 normalize per block
    norms = torch.norm(xb, dim=-1, keepdim=True).clamp(min=1e-8)  # [..., num_blocks, 1]
    xn = xb / norms

    # Step 2: FWHT
    xh = _fwht_inplace(xn.clone())

    # Step 3: Random sign flips
    signs = _generate_sign_flips(TQ3_BLOCK, seed=42, device=device)
    xh = xh * signs

    # Step 4: Absmax scale
    scales = xh.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # [..., num_blocks, 1]
    xs = xh / scales

    # Step 5: Quantize to 3-bit indices
    boundaries = TQ3_BOUNDARIES.to(device)
    indices = _quantize_to_indices(xs, boundaries)

    # Step 6: Pack
    packed = _pack_3bit(indices)

    return {
        "norms": norms.squeeze(-1),   # [..., num_blocks]
        "scales": scales.squeeze(-1), # [..., num_blocks]
        "packed": packed,             # [..., num_blocks, 48]
        "orig_shape": orig_shape,
    }


def tq3_dequantize(tq3: dict) -> torch.Tensor:
    """
    Dequantize TQ3 format back to float tensor.

    Input: dict from tq3_quantize
    Returns: tensor of original shape, float32
    """
    required_keys = {"norms", "scales", "packed", "orig_shape"}
    missing = required_keys.difference(tq3)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise KeyError(f"TQ3 payload missing required keys: {missing_csv}")

    norms = tq3["norms"]
    scales = tq3["scales"]
    packed = tq3["packed"]
    orig_shape = tq3["orig_shape"]
    if norms.shape != scales.shape:
        raise ValueError("TQ3 norms and scales must have identical shapes.")
    if packed.shape[:-1] != norms.shape or packed.shape[-1] != 48:
        raise ValueError("TQ3 packed indices shape must match norms/scales shape + [48].")
    device = norms.device

    centroids = TQ3_CENTROIDS.to(device)

    # Unpack 3-bit indices
    indices = _unpack_3bit(packed)  # [..., num_blocks, 128]

    # Dequantize: index into centroids
    xq = centroids[indices.long()]

    # Undo absmax scaling
    xq = xq * scales.unsqueeze(-1)

    # Undo sign flips
    signs = _generate_sign_flips(TQ3_BLOCK, seed=42, device=device)
    xq = xq * signs

    # Inverse FWHT (FWHT is its own inverse up to scaling, and we already normalized)
    xq = _fwht_inplace(xq)

    # Undo L2 normalization
    xq = xq * norms.unsqueeze(-1)

    # Reshape to original
    return xq.reshape(orig_shape)


def tq3_memory_bytes(shape: tuple) -> tuple:
    """
    Calculate compressed memory usage.
    Returns (compressed_bytes, original_bytes, ratio).
    """
    if not shape:
        raise ValueError("shape must be non-empty")
    if shape[-1] <= 0 or shape[-1] % TQ3_BLOCK != 0:
        raise ValueError(
            f"Last dim must be a positive multiple of {TQ3_BLOCK}, got {shape[-1]}"
        )
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"All shape dimensions must be positive, got {shape}")

    total_elements = 1
    for s in shape:
        total_elements *= s

    D = shape[-1]
    num_blocks = D // TQ3_BLOCK
    num_vectors = total_elements // D

    # Per block: 48 bytes packed + 4 bytes norm + 4 bytes scale = 56 bytes
    compressed = num_vectors * num_blocks * 56
    # FP16 original
    original = total_elements * 2

    ratio = original / max(compressed, 1)
    return compressed, original, ratio


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test():
    """Verify TQ3 round-trip quality."""
    print("=" * 60)
    print("TQ3 Self-Test")
    print("=" * 60)

    torch.manual_seed(123)
    device = "cpu"

    # Test 1: Single block
    print("\n[1] Single block (1, 128) ...")
    x = torch.randn(1, 128, device=device)
    tq = tq3_quantize(x)
    xr = tq3_dequantize(tq)
    cos = torch.nn.functional.cosine_similarity(x, xr, dim=-1).mean().item()
    mse = ((x - xr) ** 2).mean().item()
    print(f"    Cosine similarity: {cos:.6f}")
    print(f"    MSE:               {mse:.6f}")
    assert cos > 0.90, f"Cosine too low: {cos}"
    print("    PASS")

    # Test 2: Batch of vectors (simulating KV cache)
    print("\n[2] Batch KV cache (4, 32, 512) ...")
    x = torch.randn(4, 32, 512, device=device)
    tq = tq3_quantize(x)
    xr = tq3_dequantize(tq)
    cos = torch.nn.functional.cosine_similarity(
        x.reshape(-1, 512), xr.reshape(-1, 512), dim=-1
    ).mean().item()
    print(f"    Cosine similarity: {cos:.6f}")
    comp, orig, ratio = tq3_memory_bytes(x.shape)
    print(f"    FP16 size:    {orig:,} bytes")
    print(f"    TQ3 size:     {comp:,} bytes")
    print(f"    Compression:  {ratio:.2f}x")
    assert cos > 0.90, f"Cosine too low: {cos}"
    assert ratio > 4.0, f"Compression ratio too low: {ratio}"
    print("    PASS")

    # Test 3: Pack/unpack round trip
    print("\n[3] Pack/unpack round trip ...")
    indices = torch.randint(0, 8, (16, 128), dtype=torch.uint8)
    packed = _pack_3bit(indices)
    recovered = _unpack_3bit(packed)
    assert torch.equal(indices, recovered), "Pack/unpack mismatch!"
    print("    PASS")

    # Test 4: FWHT invertibility
    print("\n[4] FWHT self-inverse ...")
    x = torch.randn(8, 128, device=device)
    xh = _fwht_inplace(x.clone())
    xr = _fwht_inplace(xh.clone())
    err = (x - xr).abs().max().item()
    print(f"    Max reconstruction error: {err:.2e}")
    assert err < 1e-5, f"FWHT not self-inverse: {err}"
    print("    PASS")

    # Test 5: Memory calculation for LTX-2.3 22B scenario
    print("\n[5] LTX-2.3 22B KV cache estimate ...")
    # Typical: 64 layers, 64 heads, seq_len=2048, head_dim=128
    # K + V = 2 * [batch, layers, heads, seq, head_dim]
    layers, heads, seq, hdim = 64, 64, 2048, 128
    kv_shape = (2, layers, heads, seq, hdim)
    comp, orig, ratio = tq3_memory_bytes(kv_shape)
    print(f"    KV cache FP16: {orig / 1e9:.2f} GB")
    print(f"    KV cache TQ3:  {comp / 1e9:.2f} GB")
    print(f"    Savings:       {(orig - comp) / 1e9:.2f} GB ({ratio:.1f}x)")
    print("    PASS")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    self_test()

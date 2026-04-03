# ComfyUI-TurboQuant

TQ3 KV cache compression for ComfyUI. Reduces attention KV cache VRAM by ~4.5x using 3-bit Lloyd-Max quantization with Fast Walsh-Hadamard Transform decorrelation.

## Motivation

LTX-2.3 22B requires 32.4GB VRAM, barely fitting on a V100 32GB. A large portion of that is the KV cache in transformer attention layers. TurboQuant compresses KV tensors from FP16 (16 bits) to TQ3 (3.5 bits effective), freeing ~5x the KV cache memory.

## Installation

```bash
cd ~/ComfyUI/custom_nodes/
ln -s ~/ComfyUI-TurboQuant .
```

## Nodes

### TurboQuant KV Patch

Patches a model's attention layers to compress K and V tensors through TQ3 quantization.

- **Input**: MODEL, enabled (bool)
- **Output**: MODEL (patched)

### TurboQuant Info

Shows compression statistics after inference.

- **Input**: MODEL
- **Output**: STRING (stats)

## How TQ3 Works

Each 128-float block is compressed to 56 bytes:

1. L2 normalize the block
2. Fast Walsh-Hadamard Transform (decorrelates values)
3. Deterministic random sign flips (spreads energy)
4. Absmax scale to [-1, +1]
5. Lloyd-Max 8-level codebook quantize (3 bits/value)
6. Pack 128 indices into 48 bytes + 4B norm + 4B scale

Round-trip cosine similarity: >0.97 on typical attention vectors.

## Self-Test

```bash
cd ~/ComfyUI-TurboQuant
python -m tq3_core
```

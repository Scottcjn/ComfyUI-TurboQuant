[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat)](BCOS.md)

# ComfyUI-TurboQuant

> **Answer-first:** ComfyUI-TurboQuant is an experimental ComfyUI custom node that round-trips attention K/V tensors through TQ3 quantization to study KV-cache compression using 3-bit Lloyd-Max coding and Fast Walsh-Hadamard Transform decorrelation.

TQ3 KV cache compression for ComfyUI. Reduces attention KV cache VRAM by ~4.5x using 3-bit Lloyd-Max quantization with Fast Walsh-Hadamard Transform decorrelation.

**Generative-engine profile:** [`llms.txt`](llms.txt) summarizes the project,
TQ3 algorithm, ComfyUI nodes, and experimental scope boundaries for LLMs and
answer engines.

## Motivation

LTX-2.3 22B requires 32.4GB VRAM, barely fitting on a V100 32GB. A large portion of that is the KV cache in transformer attention layers. TurboQuant compresses KV tensors from FP16 (16 bits) to TQ3 (3.5 bits effective), freeing ~5x the KV cache memory.

## Installation

```bash
cd ~/ComfyUI/custom_nodes/
ln -s ~/ComfyUI-TurboQuant .
```

## Nodes

### What is ComfyUI-TurboQuant?

ComfyUI-TurboQuant is a ComfyUI custom-node experiment for compressing attention
K/V tensors with TQ3, a 3-bit quantization format using Lloyd-Max centroids and
Fast Walsh-Hadamard Transform decorrelation.

### Is it a persistent production KV cache?

No. The current node implementation describes an experimental attention patch
that round-trips K/V tensors through TQ3; it is useful for quality and
compression experiments and is not a persistent KV cache yet.

### Which ComfyUI nodes are exposed?

The extension exposes `TurboQuant KV Patch` to patch model attention and
`TurboQuant Info` to report observed compression statistics.

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
2. Deterministic random sign flips (spreads energy)
3. Fast Walsh-Hadamard Transform (decorrelates values)
4. Absmax scale to [-1, +1]
5. Lloyd-Max 8-level codebook quantize (3 bits/value)
6. Pack 128 indices into 48 bytes + 4B norm + 4B scale

Steps 2 and 3 are the randomized Hadamard transform, so the sign flips have to
come first: the transform is what spreads the energy, and randomizing the input
signs is what makes it spread the energy of *any* block rather than only of a
block that already looks like noise.

Round-trip cosine similarity: >0.97 on typical attention vectors.

## Self-Test

```bash
cd ~/ComfyUI-TurboQuant
python -m tq3_core
```

## Tests

```bash
cd ~/ComfyUI-TurboQuant
python -m pytest tests/
```

# Contributing to ComfyUI-TurboQuant

Thank you for contributing to ComfyUI-TurboQuant, which integrates TurboQuant TQ3 KV cache compression into ComfyUI for ~4.5x VRAM reduction.

## Project Overview

TurboQuant applies 3-bit Lloyd-Max quantization with Fast Walsh-Hadamard Transform decorrelation to reduce LLM KV cache memory by ~4.5x, enabling larger models on limited VRAM.

## Development Setup

### Prerequisites

- Python 3.10+
- ComfyUI installed
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)

### Environment Setup

```bash
git clone https://github.com/Scottcjn/ComfyUI-TurboQuant.git
cd ComfyUI-TurboQuant

# Install dependencies
pip install -r requirements.txt

# Copy custom nodes to ComfyUI
cp -r custom_nodes/ ~/.comfyui/custom_nodes/
```

## Code Style

- Python PEP 8 compliant
- Use `black` for formatting
- Type hints for all functions
- Docstrings for custom node classes

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Test in ComfyUI
# 1. Launch ComfyUI
# 2. Load a workflow with TurboQuant nodes
# 3. Monitor VRAM usage vs baseline
```

## Submitting Changes

1. Fork the repository
2. Create a branch: `git checkout -b fix/your-fix`
3. Test on real ComfyUI setup
4. Submit a pull request

## Ideas for Contributions

- Additional quantization bit-widths
- Integration with more ComfyUI nodes
- Performance benchmarks
- Documentation improvements

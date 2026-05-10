# Contributing

Thanks for helping improve ComfyUI-TurboQuant. This project is experimental
ComfyUI node code for TQ3 KV cache compression, so changes should be small,
measurable, and honest about quality and VRAM trade-offs.

## Getting Started

1. Read `README.md` to understand the current node behavior and self-test.
2. Install the repo as a ComfyUI custom node during integration testing:

   ```bash
   cd ~/ComfyUI/custom_nodes/
   ln -s ~/ComfyUI-TurboQuant .
   ```

3. Work on a focused branch:

   ```bash
   git checkout -b your-change-name
   ```

## Development Workflow

Keep changes scoped to one area:

- `tq3_core.py` for quantization, packing, dequantization, and memory math.
- `turboquant_nodes.py` for ComfyUI node behavior and model patching.
- `__init__.py` for node registration exports.
- `README.md` for usage, installation, and validation notes.

Avoid mixing algorithm changes with broad documentation cleanup. Compression
quality regressions can be subtle, so reviewers need a focused diff and clear
test evidence.

## Validation

Run the built-in self-test for core algorithm changes:

```bash
python -m tq3_core
```

For ComfyUI node changes, include:

- ComfyUI version or commit.
- Python and PyTorch versions.
- GPU model and VRAM amount when applicable.
- Workflow/model used for testing.
- Observed compression stats from the TurboQuant Info node.
- Any visible quality issues, crashes, or memory regressions.

If you cannot test inside ComfyUI, say so clearly and include the static or unit
validation you did run.

## Code Style

- Prefer simple, readable PyTorch operations over clever micro-optimizations.
- Keep tensor shape checks explicit and fail with clear error messages.
- Preserve deterministic behavior for quantization tests.
- Do not claim persistent KV-cache savings unless the implementation actually
  stores persistent cache entries in compressed form.
- Keep comments focused on non-obvious tensor layout, packing, or ComfyUI patch
  behavior.

## Pull Request Checklist

Before opening a PR, include:

- A short summary of the compression or node behavior affected.
- Commands run and their output.
- Hardware/software environment used for validation.
- Expected impact on VRAM, speed, and output quality.
- Known limitations or follow-up work.


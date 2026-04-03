"""
ComfyUI-TurboQuant: TQ3 KV Cache Compression

Reduces attention KV cache VRAM by ~4.5x using 3-bit Lloyd-Max
quantization with Fast Walsh-Hadamard Transform decorrelation.

Nodes:
  - TurboQuant KV Patch: Patches any model for compressed KV cache
  - TurboQuant Info: Shows compression stats
"""

from .turboquant_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None

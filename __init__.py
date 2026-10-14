"""Happyin Open Nodes: the open part of the Happyin ComfyUI node set.

Adding a node: put its module in nodes/, then add one line to each mapping below.
Node keys match the private Happyin pack, so existing workflows load unchanged.
"""
from .nodes.happyin_canny import HappyinCanny

NODE_CLASS_MAPPINGS = {
    "Happyin_Canny": HappyinCanny,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Happyin_Canny": "Happyin Canny",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

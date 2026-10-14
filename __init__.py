"""ComfyUI-Happyin-Canny: find the high-detail regions of an image from edge density."""
from .happyin_canny import HappyinCanny

# Same node key as in the Happyin node pack, so existing workflows keep loading.
NODE_CLASS_MAPPINGS = {"Happyin_Canny": HappyinCanny}
NODE_DISPLAY_NAME_MAPPINGS = {"Happyin_Canny": "Happyin Canny"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

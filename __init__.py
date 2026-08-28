# -*- coding: utf-8 -*-
"""
BSAI H3 UPSCAL 4K — ComfyUI custom node package entry.
视频超分放大插件入口（MiniMax H3 专属）。
"""
import os

import folder_paths

from .bsai_h3_upscale_4k import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Register bundled example workflows into ComfyUI's Workflow menu.
_WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "workflows")
try:
    if os.path.isdir(_WORKFLOWS_DIR):
        folder_paths.add_model_folder_path("workflows", _WORKFLOWS_DIR)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]



# -*- coding: utf-8 -*-
"""
BSAI H3 UPSCAL 4K — ComfyUI custom node package entry.
视频超分放大插件入口（MiniMax H3 专属）。
"""
import os

import folder_paths

from .bsai_h3_upscale_4k import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 前端扩展: web/bsai_upscale_legacy_migrate.js (旧工作流 widget 顺序迁移)
WEB_DIRECTORY = "./web"

# Register bundled example workflows into ComfyUI's Workflow menu.
_WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "workflows")
try:
    if os.path.isdir(_WORKFLOWS_DIR):
        folder_paths.add_model_folder_path("workflows", _WORKFLOWS_DIR)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]



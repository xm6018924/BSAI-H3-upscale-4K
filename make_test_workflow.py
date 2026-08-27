# -*- coding: utf-8 -*-
"""Generate the BSAI H3 upscale 4K test workflow from the official H3 t2v template."""
import json
import copy

SRC = r"C:\BSAI\ComfyUI-BSAI_pro_v38_Film Factory\ComfyUI\user\comfytv\workflows\video\local-minimax-h3-t2v.json"
DST = r"C:\BSAI\ComfyUI-BSAI_pro_v38_Film Factory\ComfyUI\user\default\workflows\BSAI H3 upscale 4K 高清放大测试工作流.json"

d = json.load(open(SRC, encoding="utf-8"))
nodes = d["nodes"]
links = d["links"]
by_id = {n["id"]: n for n in nodes}
max_id = max(n["id"] for n in nodes)
max_link = max(l[0] for l in links)

# --- new node ids / link ids ---
UP = max_id + 1          # BSAI_H3_Upscale4K
CV = max_id + 2          # CreateVideo (4K)
SV = max_id + 3          # SaveVideo (4K)
NOTE = max_id + 4        # MarkdownNote (instructions)
L_UP_IN = max_link + 1   # 122.IMAGE -> UP.images
L_UP_OUT = max_link + 2  # UP.IMAGE -> CV.images
L_AUD = max_link + 3     # 121.AUDIO -> CV.audio
L_VID = max_link + 4     # CV.VIDEO -> SV.video

# --- 1. route VAEDecode(122).IMAGE to also feed the upscaler ---
n122 = by_id[122]
n122["outputs"][0]["links"].append(L_UP_IN)  # links: [238, L_UP_IN]

# --- 2. BSAI H3 upscale 4K node ---
upscale_node = {
    "id": UP,
    "type": "BSAI_H3_Upscale4K",
    "pos": [145, 4470],
    "size": [360, 210],
    "flags": {},
    "order": 20,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": L_UP_IN}],
    "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [L_UP_OUT], "slot_index": 0},
        {"name": "width", "type": "INT", "links": None, "slot_index": 1},
        {"name": "height", "type": "INT", "links": None, "slot_index": 2},
        {"name": "scale_used", "type": "FLOAT", "links": None, "slot_index": 3},
        {"name": "info", "type": "STRING", "links": None, "slot_index": 4},
    ],
    "properties": {"Node name for S&R": "BSAI_H3_Upscale4K"},
    "widgets_values": ["RealESRGAN_x4plus.pth", 4, 256, 16, 4, True],
}
nodes.append(upscale_node)

# --- 3. CreateVideo (4K path) ---
n130 = by_id[130]
cv_node = copy.deepcopy(n130)
cv_node["id"] = CV
cv_node["pos"] = [600, 4470]
cv_node["inputs"] = [
    {"name": "images", "type": "IMAGE", "link": L_UP_OUT},
    {"name": "audio", "shape": 7, "type": "AUDIO", "link": L_AUD},
]
cv_node["outputs"] = [{"name": "VIDEO", "type": "VIDEO", "links": [L_VID]}]
nodes.append(cv_node)

# --- 4. SaveVideo (4K path) ---
n92 = by_id[92]
sv_node = copy.deepcopy(n92)
sv_node["id"] = SV
sv_node["pos"] = [980, 4490]
sv_node["inputs"] = [{"name": "video", "type": "VIDEO", "link": L_VID}]
sv_node["outputs"] = [{"name": "video", "type": "VIDEO", "links": None}]
sv_node["widgets_values"] = ["video/MiniMax_H3_4K", "auto", "auto"]
nodes.append(sv_node)

# --- 5. MarkdownNote instructions ---
note_text = (
    "## 🎬 BSAI H3 upscale 4K — 高清放大测试工作流\n\n"
    "**使用方法 / How to use:**\n"
    "1. 调整上方 H3 提示词（MiniMaxH3ImageToVideo 节点）\n"
    "2. 点「运行」Run\n\n"
    "**输出 / Outputs:**\n"
    "• 原始视频 → `output/video/MiniMax_H3`（上方 SaveVideo）\n"
    "• **4K 放大视频 → `output/video/MiniMax_H3_4K`（下方 SaveVideo）**\n\n"
    "**放大参数 / Upscale params（BSAI H3 upscale 4K 节点）:**\n"
    "• model: `RealESRGAN_x4plus.pth`（写实，可换 `x4plus_anime_6B` 极速）\n"
    "• scale: 4x ｜ tile: 256 ｜ fp16: on ｜ batch: 4\n"
    "• info 输出含实际分辨率与耗时\n\n"
    "**对比 / Compare:** 同一段 H3 视频，同时保存原始与 4K 两个版本，直接对比放大效果。"
)
note_node = {
    "id": NOTE,
    "type": "MarkdownNote",
    "pos": [-1590, 4460],
    "size": [560, 700],
    "flags": {},
    "order": 21,
    "mode": 0,
    "inputs": [],
    "outputs": [],
    "title": "BSAI H3 upscale 4K — 使用说明",
    "properties": {},
    "widgets_values": [note_text],
}
nodes.append(note_node)

# --- 6. new links ---
links.append([L_UP_IN, 122, 0, UP, 0, "IMAGE"])
links.append([L_UP_OUT, UP, 0, CV, 0, "IMAGE"])
links.append([L_AUD, 121, 0, CV, 1, "AUDIO"])
links.append([L_VID, CV, 0, SV, 0, "VIDEO"])

# --- 7. recompute order field for new nodes & write ---
json.dump(d, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("WRITTEN:", DST)
print("nodes:", len(nodes), "| links:", len(links))
print("UP node id:", UP, "| CV:", CV, "| SV:", SV, "| NOTE:", NOTE)

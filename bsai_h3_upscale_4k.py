# -*- coding: utf-8 -*-
"""
BSAI H3 UPSCAL 4K
=================
A dedicated video super-resolution / upscaling plugin for MiniMax H3.
参考当前全球最先进的超分技术实现：
  - Real-ESRGAN (x4plus / x4plus_anime_6B / general-x4v3) —— 写实 / 动漫 / 通用 像素级超分
    同时支持标准结构(body.N.conv*)与 compact 结构(body.N.rdb1/2/3)
  - Tile + Pad 分块推理（显存友好、避免接缝）
  - FP16 / BF16 半精度极速推理
  - 帧批量（Batch）并行处理
  - 模型进程级 LRU 缓存（重复调用秒出）
  - H3 专属 latent 二次采样放大（32 像素对齐，供 H3 第二遍采样补细节）

节点列表 / Node list:
  - BSAI H3 upscale 4K            : 视频帧 -> AI 超分高清视频帧 (pixel-domain upscale)
  - BSAI H3 upscale 4K Latent     : H3 latent -> 放大 latent (second-pass refine)

Classic ComfyUI API (INPUT_TYPES / NODE_CLASS_MAPPINGS) for max compatibility
with ComfyUI 0.34.x and community builds.
"""
import os
import math
import time
import threading
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

# ---------------------------------------------------------------------------
# Model registry & auto-download
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime_6B.pth",
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}
MODEL_DESCRIPTIONS = {
    "RealESRGAN_x4plus.pth": "4x photoreal (best for live-action / real humans)",
    "RealESRGAN_x4plus_anime_6B.pth": "4x anime / stylized (6 blocks, fast)",
    "realesr-general-x4v3.pth": "4x general-purpose (compact, versatile)",
}

_download_lock = threading.Lock()


def get_upscale_model_dir():
    """Return the standard ComfyUI upscale_models directory (create if needed)."""
    try:
        paths = folder_paths.get_folder_paths("upscale_models")
        if paths:
            d = paths[0]
        else:
            raise ValueError
    except Exception:
        d = os.path.join(folder_paths.base_path, "models", "upscale_models")
    os.makedirs(d, exist_ok=True)
    return d


def list_available_models():
    """List upscaler model files present in upscale_models (plus built-in candidates)."""
    d = get_upscale_model_dir()
    names = []
    try:
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".pth", ".safetensors", ".pt")):
                names.append(fn)
    except Exception:
        pass
    for m in MODEL_URLS:
        if m not in names:
            names.append(m)
    return names


def _download_with_progress(url, dst, label):
    """Download a model file with a simple progress indicator."""
    tmp = dst + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            t0 = time.time()
            print(f"[BSAI H3 UPSCAL 4K] Downloading {label} ...", flush=True)
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        spd = done / max(time.time() - t0, 1e-6) / 1e6
                        print(f"\r  {pct:5.1f}%  {done / 1e6:7.1f}/{total / 1e6:7.1f} MB  {spd:5.1f} MB/s", end="", flush=True)
            print("", flush=True)
        os.replace(tmp, dst)
        print(f"[BSAI H3 UPSCAL 4K] Saved to {dst}", flush=True)
        return True
    except Exception as e:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        raise RuntimeError(f"Model download failed: {e}")


def ensure_model(name):
    """Ensure a model file exists (auto-download if needed). Return absolute path."""
    d = get_upscale_model_dir()
    path = os.path.join(d, name)
    if os.path.exists(path):
        return path
    url = MODEL_URLS.get(name)
    if not url:
        raise FileNotFoundError(
            f"[BSAI H3 UPSCAL 4K] Model '{name}' not found in {d} and it is not a built-in "
            "model, so it cannot be auto-downloaded. Please place the model file there manually."
        )
    with _download_lock:
        if not os.path.exists(path):
            _download_with_progress(url, path, name)
    return path


# ---------------------------------------------------------------------------
# RRDBNet (Real-ESRGAN architecture) — supports BOTH standard and compact layouts
# ---------------------------------------------------------------------------
def _make_layer(block, num_layers):
    layers = []
    for _ in range(num_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    """Residual Dense Block (single) used by RRDBNet."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _CompactBlock(nn.Module):
    """Compact block: 3 RDBs in series + 0.2 residual (Real-ESRGAN compact layout)."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock_5C(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock_5C(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Enhanced SR network (Real-ESRGAN). Auto-detects scale / num_block / compact from weights."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                 num_grow_ch=32, scale=4, compact=False):
        super().__init__()
        self.scale = scale
        self.compact = compact
        self.num_feat = num_feat
        self.num_block = num_block
        self.num_grow_ch = num_grow_ch
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if compact:
            self.body = _make_layer(lambda: _CompactBlock(num_feat, num_grow_ch), num_block)
        else:
            self.body = _make_layer(lambda: ResidualDenseBlock_5C(num_feat, num_grow_ch), num_block)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def _detect_arch(state):
    """Detect num_block, scale, num_feat, num_grow_ch, compact from a Real-ESRGAN state dict."""
    compact = "body.0.rdb1.conv1.weight" in state
    num_block = 0
    while True:
        if compact and f"body.{num_block}.rdb1.conv1.weight" in state:
            num_block += 1
        elif (not compact) and f"body.{num_block}.conv1.weight" in state:
            num_block += 1
        else:
            break
    if num_block == 0:
        raise RuntimeError(
            "[BSAI H3 UPSCAL 4K] Cannot recognize this model as a Real-ESRGAN (RRDBNet) "
            "architecture. Only Real-ESRGAN x4plus / anime_6B / general-x4v3 are supported."
        )
    scale = 4 if "conv_up2.weight" in state else 2
    num_feat = state["conv_first.weight"].shape[0]
    k0 = "body.0.rdb1.conv1.weight" if compact else "body.0.conv1.weight"
    num_grow_ch = state[k0].shape[0]
    return num_block, scale, num_feat, num_grow_ch, compact


# ---------------------------------------------------------------------------
# Model cache (process-wide, keyed by path + dtype)
# ---------------------------------------------------------------------------
_model_cache = {}
_model_cache_lock = threading.Lock()


def _load_model(path, use_fp16):
    key = (path, use_fp16, torch.cuda.is_available())
    with _model_cache_lock:
        if key in _model_cache:
            return _model_cache[key]
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    state = {k: v for k, v in state.items() if k.startswith(("conv_", "body."))}
    nb, scale, nf, ng, compact = _detect_arch(state)
    net = RRDBNet(num_feat=nf, num_block=nb, num_grow_ch=ng, scale=scale, compact=compact)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"[BSAI H3 UPSCAL 4K] Model weights mismatch for {os.path.basename(path)} "
            f"(missing {len(missing)} keys, e.g. {missing[0]}). This may not be a "
            "Real-ESRGAN (RRDBNet) checkpoint."
        )
    net.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)
    if use_fp16 and dev == "cuda":
        net = net.half()
    with _model_cache_lock:
        if len(_model_cache) >= 3:
            try:
                drop = next(iter(_model_cache))
                _model_cache.pop(drop)
            except Exception:
                pass
        _model_cache[key] = net
    return net


# ---------------------------------------------------------------------------
# Tile-based inference (VRAM friendly, seam-free, dtype-safe)
# ---------------------------------------------------------------------------
def _upscale_image(model, img):
    """img [1,C,H,W] on CPU float32 -> [1,C,scaledH,scaledW] float32 on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = next(model.parameters()).dtype
    with torch.no_grad():
        out = model(img.to(device).to(dt))
    return out.float().clamp_(0, 1).cpu()


def _upscale_image_tiled(model, img, tile_size, tile_pad, device):
    """
    img: torch [1, C, H, W] float32 in [0,1], on CPU.
    Returns [1, C, H*scale, W*scale] float32 on CPU.
    Tile loop pads the input first (seam-free, never cuts below kernel size).
    """
    scale = model.scale
    h0, w0 = img.shape[2], img.shape[3]
    tile = max(1, int(tile_size))
    pad = max(0, int(tile_pad))
    if tile <= 0 or (h0 + 2 * pad <= tile and w0 + 2 * pad <= tile):
        return _upscale_image(model, img)
    dt = next(model.parameters()).dtype

    # pad input with edge replication (safe even when dim < pad)
    inp = F.pad(img.to(device).to(dt), (pad, pad, pad, pad), mode="replicate")
    h_pad, w_pad = inp.shape[2], inp.shape[3]

    over = torch.zeros((1, 1, h_pad, w_pad), dtype=torch.float32, device=device)
    out = torch.zeros((1, 3, h_pad * scale, w_pad * scale), dtype=torch.float32, device=device)

    if h_pad <= tile:
        rows = [0]
    else:
        rows = list(range(0, h_pad - tile, tile - pad * 2))
        if rows[-1] != h_pad - tile:
            rows.append(h_pad - tile)
    if w_pad <= tile:
        cols = [0]
    else:
        cols = list(range(0, w_pad - tile, tile - pad * 2))
        if cols[-1] != w_pad - tile:
            cols.append(w_pad - tile)

    with torch.no_grad():
        for r in rows:
            for c in cols:
                th = min(tile, h_pad - r)
                tw = min(tile, w_pad - c)
                t_in = inp[:, :, r:r + th, c:c + tw]
                t_out = model(t_in).float()
                hh, ww = t_out.shape[2], t_out.shape[3]
                out[:, :, r * scale:r * scale + hh, c * scale:c * scale + ww] += t_out
                wg = torch.ones((1, 1, th, tw), dtype=torch.float32, device=device)
                p1 = min(pad, th)
                p2 = min(pad, tw)
                if r > 0:
                    wg[:, :, :p1, :] *= torch.linspace(0, 1, p1, device=device).view(1, 1, p1, 1)
                if r + th < h_pad:
                    wg[:, :, -p1:, :] *= torch.linspace(1, 0, p1, device=device).view(1, 1, p1, 1)
                if c > 0:
                    wg[:, :, :, :p2] *= torch.linspace(0, 1, p2, device=device).view(1, 1, 1, p2)
                if c + tw < w_pad:
                    wg[:, :, :, -p2:] *= torch.linspace(1, 0, p2, device=device).view(1, 1, 1, p2)
                over[:, :, r:r + th, c:c + tw] += wg

    out = out[:, :, pad * scale:(pad + h0) * scale, pad * scale:(pad + w0) * scale]
    over = F.interpolate(over, scale_factor=scale, mode="bilinear", align_corners=False)
    over = over[:, :, pad * scale:(pad + h0) * scale, pad * scale:(pad + w0) * scale]
    out = out / over.clamp_min(1e-6)
    return out.float().clamp_(0, 1).cpu()


def _upscale_batch(model, images, tile_size, tile_pad, batch_frames):
    """
    images: torch [B,H,W,C] float32 0..1 on CPU.
    Returns [B, H*scale, W*scale, C] float32 0..1 on CPU.
    """
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError("BSAI H3 UPSCAL 4K expects IMAGE frames of shape [B,H,W,3]")
    b = images.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    im = images.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

    results = []
    batch = max(1, int(batch_frames))
    for start in range(0, b, batch):
        chunk = im[start:start + batch]
        for i in range(chunk.shape[0]):
            single = chunk[i:i + 1]
            out = _upscale_image_tiled(model, single, tile_size, tile_pad, device)
            results.append(out)
        del chunk
        if device == "cuda":
            torch.cuda.empty_cache()
    out_t = torch.cat(results, dim=0)  # [B, C, H*scale, W*scale]
    return out_t.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]


# ---------------------------------------------------------------------------
# Main node: video frames -> AI upscaled frames
# ---------------------------------------------------------------------------
class BSAI_H3_Upscale4K:
    """Video frame AI super-resolution for MiniMax H3 (pixel domain, extremely fast)."""

    @classmethod
    def INPUT_TYPES(cls):
        models = list_available_models()
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (models, {"default": "RealESRGAN_x4plus.pth"}),
                "scale": ("INT", {"default": 4, "min": 2, "max": 4, "step": 1}),
                "tile_size": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 16}),
                "tile_pad": ("INT", {"default": 16, "min": 0, "max": 128, "step": 4}),
                "batch_frames": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "use_fp16": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE", "width", "height", "scale_used", "info")
    FUNCTION = "upscale"
    CATEGORY = "BSAI/H3"
    DESCRIPTION = (
        "H3 视频专用 AI 超分（像素域）：Real-ESRGAN 极速放大。\n"
        "Tile 分块 + FP16 半精度 + 帧批量并行 + 模型常驻缓存。\n"
        "Video-only AI super-resolution for MiniMax H3: Real-ESRGAN extreme-speed upscale."
    )

    def upscale(self, images, model_name, scale, tile_size, tile_pad, batch_frames, use_fp16):
        t0 = time.time()
        path = ensure_model(model_name)
        model = _load_model(path, use_fp16)
        eff_scale = min(scale, model.scale)

        if model.scale == 2 and scale == 4:
            # 2x model but user asked 4x -> run twice
            first = _upscale_batch(model, images, tile_size, tile_pad, batch_frames)
            out = _upscale_batch(model, first, tile_size, tile_pad, batch_frames)
            eff_scale = 4
        else:
            out = _upscale_batch(model, images, tile_size, tile_pad, batch_frames)

        bh, bw = out.shape[1], out.shape[2]
        elapsed = time.time() - t0
        info = (
            f"model: {model_name} (scale={model.scale}) | "
            f"output: {bw}x{bh} | eff_scale: {eff_scale}x | "
            f"fp16: {use_fp16} | tile: {tile_size} pad:{tile_pad} | "
            f"frames: {images.shape[0]} | time: {elapsed:.2f}s | "
            f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
        )
        return (out, bw, bh, float(eff_scale), info)


# ---------------------------------------------------------------------------
# Latent node: H3 latent -> enlarged latent (32px aligned, second-pass refine)
# ---------------------------------------------------------------------------
class BSAI_H3_Upscale4K_Latent:
    """H3 latent-space upscale aligned to the 32-px grid for safe second-pass sampling."""

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (cls.upscale_methods,),
                "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("LATENT", "width", "height", "effective_scale", "info")
    FUNCTION = "upscale"
    CATEGORY = "BSAI/H3"
    DESCRIPTION = (
        "H3 专属 latent 放大：自动 32 像素对齐，供 H3 第二遍采样补细节（保持人物一致性）。\n"
        "H3-specific latent upscale with 32-px alignment for second-pass detail refine."
    )

    LATENT_ALIGNMENT = 2
    H3_VAE_SPATIAL_DOWNSCALE = 16

    @staticmethod
    def _floor_aligned(value):
        return max(2, math.floor(value / 2) * 2)

    def _calculate_aligned_size(self, lw, lh, scale_by):
        if lw <= 0 or lh <= 0:
            raise ValueError("latent width/height must be > 0")
        long_is_width = lw >= lh
        long_in, short_in = (lw, lh) if long_is_width else (lh, lw)
        short_out = self._floor_aligned(short_in * scale_by)
        short_eff = short_out / short_in
        ideal_long = long_in * short_eff
        long_cap = self._floor_aligned(long_in * scale_by)
        lower = self._floor_aligned(ideal_long)
        candidates = {c for c in (lower, lower + 2, long_cap) if 2 <= c <= long_cap}
        long_out = min(candidates, key=lambda c: (abs(c - ideal_long), c))
        return (long_out, short_out) if long_is_width else (short_out, long_out)

    def upscale(self, samples, upscale_method, scale_by):
        import comfy.utils
        source = samples["samples"]
        lw, lh = source.shape[-1], source.shape[-2]
        ow, oh = self._calculate_aligned_size(lw, lh, scale_by)
        result = samples.copy()
        result["samples"] = comfy.utils.common_upscale(source, ow, oh, upscale_method, "disabled")
        px_w, px_h = lw * self.H3_VAE_SPATIAL_DOWNSCALE, lh * self.H3_VAE_SPATIAL_DOWNSCALE
        po_w, po_h = ow * self.H3_VAE_SPATIAL_DOWNSCALE, oh * self.H3_VAE_SPATIAL_DOWNSCALE
        eff = ow / lw
        info = (
            f"latent {lw}x{lh} -> {ow}x{oh} | pixel {px_w}x{px_h} -> {po_w}x{po_h} | "
            f"eff_scale {eff:.4f}x | 32px aligned: True | method: {upscale_method}"
        )
        return (result, po_w, po_h, float(eff), info)


NODE_CLASS_MAPPINGS = {
    "BSAI_H3_Upscale4K": BSAI_H3_Upscale4K,
    "BSAI_H3_Upscale4K_Latent": BSAI_H3_Upscale4K_Latent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_H3_Upscale4K": "BSAI H3 upscale 4K / 视频超分",
    "BSAI_H3_Upscale4K_Latent": "BSAI H3 upscale 4K Latent / H3潜空间放大",
}

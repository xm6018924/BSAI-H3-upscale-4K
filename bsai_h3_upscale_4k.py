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
import sys
import math
import time
import threading
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

import folder_paths

# Speed: let cuDNN autotune conv kernels for the fixed video-frame sizes used here.
# Measured ~2x faster on RTX 5090 vs default heuristics.
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

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
        # NOTE: Real-ESRGAN (x4plus/anime_6B/general-x4v3) trained WITHOUT activation
        # after conv_first. Adding LeakyReLU here shifts features into a nonlinear
        # regime and causes a global dark + red/magenta color cast (verified against
        # spandrel reference implementation). Keep conv_first activation-free.
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def _detect_arch(state):
    """
    Detect architecture + hyper-parameters from a Real-ESRGAN family state dict.

    Returns:
      ("rrdb", dict(num_block, scale, num_feat, num_grow_ch, compact)) for
              RRDBNet (x4plus / anime_6B) and its compact layout (general-x4v3
              is SRVGG though, see below).
      ("srvgg", dict(num_feat)) for SRVGGNetCompact (realesr-general-x4v3).
    """
    # --- RRDBNet (standard + compact): has conv_first + body.N.{rdb1,rdb2,rdb3} ---
    if "conv_first.weight" in state and "body.0.rdb1.conv1.weight" in state:
        compact = True
        k0 = "body.0.rdb1.conv1.weight"
    elif "conv_first.weight" in state and "body.0.conv1.weight" in state:
        compact = False
        k0 = "body.0.conv1.weight"
    else:
        compact = None
        k0 = None

    if compact is not None:
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
        num_grow_ch = state[k0].shape[0]
        return "rrdb", dict(num_block=num_block, scale=scale, num_feat=num_feat,
                            num_grow_ch=num_grow_ch, compact=compact)

    # --- SRVGGNetCompact (realesr-general-x4v3): body.N alternating Conv / PReLU ---
    if "body.0.weight" in state and "body.0.bias" in state and "body.1.weight" in state:
        num_feat = state["body.0.weight"].shape[0]
        return "srvgg", dict(num_feat=num_feat)

    raise RuntimeError(
        "[BSAI H3 UPSCAL 4K] Cannot recognize this model as a Real-ESRGAN family "
        "architecture. Only Real-ESRGAN x4plus / anime_6B / general-x4v3 are supported."
    )


class SRVGGNetCompact(nn.Module):
    """
    SRVGGNetCompact — the tiny general-purpose 4x model (realesr-general-x4v3).
    body: alternating Conv2d (even index) + PReLU (odd index), tail Conv out=3*scale^2,
    then a single PixelShuffle. Layers are reconstructed from the state dict keys.
    """

    def __init__(self, state):
        super().__init__()
        body = nn.ModuleList()
        i = 0
        while True:
            wkey = f"body.{i}.weight"
            if wkey not in state:
                break
            bkey = f"body.{i}.bias"
            w = state[wkey]
            if bkey in state:
                body.append(nn.Conv2d(w.shape[1], w.shape[0], 3, 1, 1))
            else:
                body.append(nn.PReLU(num_parameters=w.shape[0]))
            i += 1
        self.body = body
        # tail conv determines upscale factor (out == 3 * scale^2)
        tail_out = body[-1].out_channels
        self.scale = int(round(math.sqrt(max(tail_out // 3, 1)))) if tail_out % 3 == 0 else 1
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.pixel_shuffle(out)
        # SRVGGNetCompact learns the residual: add back the nearest-upsampled input.
        base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return out + base


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
    arch, params = _detect_arch(state)
    if arch == "srvgg":
        net = SRVGGNetCompact(state)
        missing, unexpected = net.load_state_dict(state, strict=True)
    else:
        net = RRDBNet(**params)
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
# torch.compile acceleration (fixed-shape video frames -> CUDA graphs / fusion)
# ---------------------------------------------------------------------------
_compiled_cache = {}
_compiled_cache_lock = threading.Lock()


def _detect_cuda_malloc_async():
    """Detect whether the current CUDA allocator is cudaMallocAsync, which is
    incompatible with torch.instrument cudagraph_trees (raises
    'cudaMallocAsync does not yet support checkPoolLiveAllocations').
    Returns True if cudaMallocAsync is in use.
    """
    if not torch.cuda.is_available():
        return False
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "cudaMallocAsync" in conf:
        return True
    try:
        backend = torch.cuda.memory._get_allocator_backend()
        return "async" in str(backend).lower()
    except Exception:
        pass
    return False


class _CompiledWrapper(nn.Module):
    """Expose model.scale/parameters while running the torch.compile graph."""

    def __init__(self, model, mode="reduce-overhead"):
        super().__init__()
        self._m = model
        self.scale = model.scale
        self.num_block = getattr(model, "num_block", None)
        self._compiled = torch.compile(model, mode=mode, dynamic=False)

    def forward(self, x):
        return self._compiled(x)

    def parameters(self, recurse=True):
        return self._m.parameters(recurse)


def _load_model_compiled(path, use_fp16):
    """Load model (from cache) and return a torch.compile-wrapped fast version."""
    key = (path, use_fp16, torch.cuda.is_available())
    with _compiled_cache_lock:
        if key in _compiled_cache:
            return _compiled_cache[key]
    try:
        model = _load_model(path, use_fp16)
        # cudaMallocAsync compatibility: cudagraph_trees calls
        # checkPoolLiveAllocations which is unsupported on cudaMallocAsync.
        # Disable cudagraph_trees and fall back to mode="default" (no CUDA graph)
        # when cudaMallocAsync is detected (e.g. ComfyUI started without
        # --disable-cuda-malloc on RTX 50xx + cu130).
        if _detect_cuda_malloc_async():
            try:
                torch._inductor.config.triton.cudagraph_trees = False
            except Exception:
                pass
            print("[BSAI H3 UPSCAL 4K] cudaMallocAsync detected: using torch.compile "
                  "mode='default' (cudagraph disabled) for compatibility.", flush=True)
            wrapped = _CompiledWrapper(model, mode="default")
        else:
            # NOTE: no dummy warmup here. torch.compile captures the CUDA graph on
            # the FIRST real call with the actual video-frame shape; a differently-
            # shaped dummy would force a re-compile later (slower). First real run
            # pays the one-time compile cost, then every subsequent same-shape run
            # is fast.
            wrapped = _CompiledWrapper(model)
    except Exception as e:
        print(f"[BSAI H3 UPSCAL 4K] torch.compile unavailable, falling back to eager "
              f"({type(e).__name__}: {e})", flush=True)
        wrapped = model
    with _compiled_cache_lock:
        if len(_compiled_cache) >= 2:
            try:
                _compiled_cache.pop(next(iter(_compiled_cache)))
            except Exception:
                pass
        _compiled_cache[key] = wrapped
    return wrapped


# ---------------------------------------------------------------------------
# Tile-based inference (VRAM friendly, seam-free, dtype-safe)
# ---------------------------------------------------------------------------
def _upscale_image(model, img):
    """img [1,C,H,W] on CPU float32 -> [1,C,scaledH,scaledW] float32 (stays on GPU)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = next(model.parameters()).dtype
    with torch.no_grad():
        out = model(img.to(device).to(dt))
    return out.float().clamp_(0, 1)


def _upscale_image_tiled(model, img, tile_size, tile_pad, device):
    """
    img: torch [1, C, H, W] float32 in [0,1], on CPU.
    Returns [1, C, H*scale, W*scale] float32 in [0,1] (stays on GPU).
    Tile loop pads the input first (seam-free, never cuts below kernel size).
    """
    scale = model.scale
    h0, w0 = img.shape[2], img.shape[3]
    tile = int(tile_size)  # <=0  => full-image (no tiling, fastest on big-VRAM GPUs)
    pad = max(0, int(tile_pad))
    if tile <= 0 or (h0 + 2 * pad <= tile and w0 + 2 * pad <= tile):
        return _upscale_image(model, img)
    dt = next(model.parameters()).dtype

    # pad input with edge replication (safe even when dim < pad)
    inp = F.pad(img.to(device).to(dt), (pad, pad, pad, pad), mode="replicate")
    h_pad, w_pad = inp.shape[2], inp.shape[3]

    # overlap weight map lives in OUTPUT space (repeat_interleave == exact pixel
    # alignment with the accumulated `out`; a bilinear upscale would shift weights
    # at tile borders and cause visible block/grid seams)
    over = torch.zeros((1, 1, h_pad * scale, w_pad * scale), dtype=torch.float32, device=device)
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
                wg_out = wg.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
                # weighted accumulate: out must carry the same weights as `over`,
                # otherwise the normalized blend over tile overlaps produces
                # visible brightness bands (block/grid seams)
                out[:, :, r * scale:r * scale + th * scale, c * scale:c * scale + tw * scale] += t_out * wg_out
                over[:, :, r * scale:r * scale + th * scale, c * scale:c * scale + tw * scale] += wg_out

    out = out[:, :, pad * scale:(pad + h0) * scale, pad * scale:(pad + w0) * scale]
    over = over[:, :, pad * scale:(pad + h0) * scale, pad * scale:(pad + w0) * scale]
    out = out / over.clamp_min(1e-6)
    return out.float().clamp_(0, 1)  # stays on GPU


def _upscale_batch(model, images, tile_size, tile_pad, batch_frames):
    """
    images: torch [B,H,W,C] float32 0..1 on CPU.
    Returns [B, H*scale, W*scale, C] float32 0..1 on CPU.

    Fast path: frames are H2D-copied in chunks, then inferred ONE frame at a time
    (measured fastest on RTX 5090 — batched conv is actually *slower* for these
    small CNNs), with the compiled/eager model and NO torch.cuda.empty_cache()
    between frames (empty_cache is a huge stall and was a major slowdown).
    """
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError("BSAI H3 UPSCAL 4K expects IMAGE frames of shape [B,H,W,3]")
    b = images.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    im = images.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W] on CPU
    dt = next(model.parameters()).dtype

    results = []
    chunk = max(1, int(batch_frames))
    for start in range(0, b, chunk):
        # stage a chunk on GPU at once (fewer H2D syncs), then infer frames singly.
        # Results accumulate ON GPU within the chunk (no per-frame .cpu() stall),
        # then one .cpu() per chunk keeps VRAM bounded for long videos.
        gpu_chunk = im[start:start + chunk].to(device).to(dt)
        chunk_res = []
        for i in range(gpu_chunk.shape[0]):
            chunk_res.append(_upscale_image_tiled(model, gpu_chunk[i:i + 1], tile_size, tile_pad, device))
        del gpu_chunk
        results.append(torch.cat(chunk_res, dim=0).cpu())
    out_t = torch.cat(results, dim=0)  # [B, C, H*scale, W*scale] fp32
    return out_t.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]


# ---------------------------------------------------------------------------
# Temporal consistency + detail enhancement (motion-compensated video refine)
# ---------------------------------------------------------------------------
# The per-frame CNN upscaler is temporally independent: on video, uncorrelated
# per-frame noise makes static regions shimmer / flicker. We fix this with a
# light Farneback optical-flow step (computed on the LR frames, rescaled to SR):
# each SR frame is blended with its motion-compensated neighbours; the blend
# weight is attenuated by local motion magnitude so fast/occluded areas do not
# ghost. Then an optional separable-Gaussian unsharp-mask adds crisp detail.
# All heavy math stays on GPU; only the LR flow is computed on CPU (cheap).

def _frame_gray(frame_np):
    """frame_np [h,w,3] float32 0-1 -> grayscale uint8 for Farneback (faster)."""
    img = np.clip(frame_np, 0, 1)
    g = img.astype(np.float32).mean(axis=2)
    return (g * 255.0).astype(np.uint8)


def _compute_flow_pairs(lr_np, max_size=512):
    """
    lr_np: [B,h,w,3] float32 0-1 numpy.
    Returns list of len B-1: flows[i] = optical flow from frame i to frame i+1
    (h,w,2) float32, in *original LR* pixel coordinates (rescaled back up).
    Downscales internally to speed up Farneback; flow is upsampled to LR size.
    """
    if not _HAS_CV2:
        return None
    B = lr_np.shape[0]
    if B < 2:
        return []
    h, w = lr_np.shape[1], lr_np.shape[2]
    ds = 1.0
    th, tw = h, w
    if max(max(h, w), 1) > max_size:
        ds = max_size / float(max(h, w))
        th, tw = max(2, int(round(h * ds))), max(2, int(round(w * ds)))
    prev = cv2.resize(_frame_gray(lr_np[0]), (tw, th), interpolation=cv2.INTER_AREA)
    flows = []
    for i in range(1, B):
        cur = cv2.resize(_frame_gray(lr_np[i]), (tw, th), interpolation=cv2.INTER_AREA)
        f = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if ds != 1.0:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) / ds
        flows.append(f.astype(np.float32))
        prev = cur
    return flows


_SR_GRID_CACHE = {}


def _sr_grid(H, W, device):
    """Cached normalized coordinate grid [1,H,W,2] (align_corners) for the SR size."""
    key = (H, W, str(device))
    g = _SR_GRID_CACHE.get(key)
    if g is None:
        yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing="ij")
        gx = xx.float() * 2.0 / max(W - 1, 1) - 1.0
        gy = yy.float() * 2.0 / max(H - 1, 1) - 1.0
        g = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        if len(_SR_GRID_CACHE) > 4:
            _SR_GRID_CACHE.clear()
        _SR_GRID_CACHE[key] = g
    return g


def _warp_flow(img, ft, grid):
    """
    img [1,3,H,W] cuda; ft [1,2,H,W] SR-space flow (cuda); grid cached [1,H,W,2].
    Warps img by ft (queries img at position p + ft). Returns [1,3,H,W].
    """
    H, W = img.shape[2], img.shape[3]
    fx = ft[:, 0:1] * (2.0 / max(W - 1, 1))
    fy = ft[:, 1:2] * (2.0 / max(H - 1, 1))
    g = grid.to(img.dtype) + torch.cat([fx, fy], dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)


def _flow_motion_weight(ft):
    """Per-pixel confidence: high motion (large flow) contributes less (avoids ghosting).
    ft [1,2,H,W] -> [1,1,H,W] in [0,1]."""
    mag = ft.norm(dim=1, keepdim=True)
    sig = max(ft.shape[2], ft.shape[3]) * 0.05
    return torch.exp(-mag / max(sig, 1e-3))


def _lce_gpu(x, strength):
    """Lightweight local-contrast enhancement on [n,3,H,W] cuda float.

    Method intent (aligns with Topaz / FlashVSR 'generated texture' feel):
    enhanced = x + g * (x - local_mean) with an adaptive gain g that is strong
    on mild-contrast texture and gated near strong edges (no halos). This
    *rebuilds* texture contrast instead of only sharpening. Cheap separable
    GPU blur, negligible cost on 4K video.
    """
    if strength <= 0:
        return x
    n, C, H, W = x.shape
    ks = max(15, (min(H, W) // 8) | 1)
    ks = min(ks, 129) | 1
    if ks > 3:
        half = ks // 2
        dev, dt = x.device, x.dtype
        sig = ks / 4.0
        ax = torch.arange(-half, half + 1, dtype=torch.float32, device=dev)
        g = torch.exp(-(ax * ax) / (2 * sig * sig))
        g = (g / g.sum()).to(dt)
        k1 = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        k2 = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        local = F.conv2d(x, k1, padding=(half, 0), groups=C)
        local = F.conv2d(local, k2, padding=(0, half), groups=C)
    else:
        local = x
    d = x - local
    gate = torch.sigmoid(-d.abs() * 8.0 + 1.0)  # 1 on mild texture, ~0 on edges
    return x + strength * gate * d


def _detail_enhance_gpu(frames, amount, radius, mode="classic"):
    """
    frames [n,H,W,3] or [n,3,H,W] cuda -> multi-scale detail rebuild, same
    shape/dtype. v1.9.0: replaces the single-scale unsharp with a luminance-domain
    multi-scale DoG (small + medium) gated by a local-variance mask, so texture is
    *reconstructed* (edge/tone natural) instead of merely oversharpened; flat
    areas (skin) stay untouched to avoid a plastic look. mode='smart' additionally
    runs a light local-contrast rebuild (_lce_gpu) for generative-style texture.
    """
    if amount <= 0:
        return frames
    is_chw = (frames.ndim == 4 and frames.shape[1] == 3 and frames.shape[3] != 3)
    x = frames if is_chw else frames.permute(0, 3, 1, 2)  # [n,3,H,W]
    n, C, H, W = x.shape
    dev, dt = x.device, x.dtype

    def _blur(t, sig):
        ks = int(math.ceil(sig * 4)) | 1
        half = ks // 2
        ax = torch.arange(-half, half + 1, dtype=torch.float32, device=dev)
        g = torch.exp(-(ax * ax) / (2 * sig * sig))
        g = (g / g.sum()).to(dt)
        k1 = g.view(1, 1, -1, 1).repeat(t.shape[1], 1, 1, 1)
        k2 = g.view(1, 1, 1, -1).repeat(t.shape[1], 1, 1, 1)
        b = F.conv2d(t, k1, padding=(half, 0), groups=t.shape[1])
        b = F.conv2d(b, k2, padding=(0, half), groups=t.shape[1])
        return b

    # luminance (Rec.601) as the single detail-carrying plane
    y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]  # [n,1,H,W]
    rs = max(0.5, float(radius))
    rm = rs * 2.4
    yb1 = _blur(y, rs)   # small scale
    yb2 = _blur(y, rm)   # medium scale
    d_small = y - yb1    # micro detail / edges
    d_mid = yb1 - yb2    # mid-frequency texture
    # local-variance gate: strong in textured/edge areas, weak on flat skin
    lv = F.avg_pool2d(d_small.abs(), kernel_size=9, stride=1, padding=4)
    vmax = lv.amax(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    w = (lv / vmax).clamp(0.25, 1.0)
    det = (0.7 * d_small + 0.45 * d_mid) * w
    det = torch.clamp(det, -0.25, 0.25)
    out = x + amount * det  # luminance-detail added back to all channels (no color fringing)
    if mode == "smart":
        out = _lce_gpu(out, min(amount * 0.7, 0.40))
    return out if is_chw else out.permute(0, 2, 3, 1)


def _soften_gpu(frames, softness, radius=1.2):
    """Light softening blend to balance sharpening overshoot / blocky artifacts.

    Method borrowed from Topaz Starlight's "softness" control (its 星光 model ships
    with softness=1): after the detail-enhancement pass, blend a small fraction of
    a Gaussian-blurred copy back in. frames [n,H,W,3] or [n,3,H,W]; returns same
    layout/dtype/device. GPU separable Gaussian, reuse the same kernel builder.
    """
    if softness <= 0:
        return frames
    is_chw = (frames.ndim == 4 and frames.shape[1] == 3 and frames.shape[3] != 3)
    x = frames if is_chw else frames.permute(0, 3, 1, 2)
    n, C, H, W = x.shape
    sig = max(0.5, float(radius))
    ks = int(math.ceil(sig * 4)) | 1
    half = ks // 2
    dev, dt = x.device, x.dtype
    ax = torch.arange(-half, half + 1, dtype=torch.float32, device=dev)
    g = torch.exp(-(ax * ax) / (2 * sig * sig))
    g = (g / g.sum()).to(dt)
    k1 = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    k2 = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    xb = F.conv2d(x, k1, padding=(half, 0), groups=C)
    xb = F.conv2d(xb, k2, padding=(0, half), groups=C)
    out = x * (1.0 - softness) + xb * softness
    return out if is_chw else out.permute(0, 2, 3, 1)


def _video_temporal_detail(sr_cpu, lr_np, temporal_strength, detail_amount, detail_radius, scale, detail_mode="classic"):
    """
    sr_cpu: [B,H,W,3] float32 CPU 0-1 (already upscaled).
    lr_np:  [B,h,w,3] float32 0-1 numpy (original LR frames, for optical flow).
    Returns fused + enhanced frames [B,H,W,3] CPU float32.

    Fast path: frames are staged to GPU in windows (with neighbours) in ONE
    transfer, all math runs in fp16, and the warp grid is cached per (H,W).
    """
    B = sr_cpu.shape[0]
    if (temporal_strength <= 0 and detail_amount <= 0) or B < 1:
        return sr_cpu
    flows = _compute_flow_pairs(lr_np) if temporal_strength > 0 else None
    H, W = sr_cpu.shape[1], sr_cpu.shape[2]
    use_gpu = torch.cuda.is_available()
    dev = torch.cuda.current_device() if use_gpu else torch.device("cpu")
    half = use_gpu
    pix = H * W
    window = 8 if pix > 10_000_000 else 16
    grid = _sr_grid(H, W, dev) if flows is not None else None
    out_chunks = []
    for s in range(0, B, window):
        e = min(B, s + window)
        lo = max(0, s - 1)
        hi = min(B, e + 1)
        buf = sr_cpu[lo:hi].to(dev).permute(0, 3, 1, 2)  # [m,3,H,W] one transfer
        if half:
            buf = buf.half()
        rel = {i: i - lo for i in range(lo, hi)}
        chunk = buf[rel[s]:rel[e - 1] + 1].clone()
        if flows is not None:
            n = chunk.shape[0]
            fused = chunk.clone()
            for j in range(n):
                i = s + j
                acc = chunk[j:j + 1].clone()
                denom = torch.ones(1, 1, H, W, device=dev, dtype=acc.dtype)
                if i > 0:
                    ft = torch.from_numpy(flows[i - 1]).float().to(dev)
                    ft = F.interpolate(ft.permute(2, 0, 1).unsqueeze(0) * scale,
                                       size=(H, W), mode="bilinear", align_corners=False)
                    if half:
                        ft = ft.half()
                    wimg = _warp_flow(buf[rel[i - 1]:rel[i - 1] + 1], ft, grid)
                    w = (temporal_strength * _flow_motion_weight(ft)).to(acc.dtype)
                    acc = acc + w * wimg
                    denom = denom + w
                if i < B - 1:
                    ft = torch.from_numpy(flows[i]).float().to(dev)
                    ft = F.interpolate(ft.permute(2, 0, 1).unsqueeze(0) * scale,
                                       size=(H, W), mode="bilinear", align_corners=False)
                    if half:
                        ft = ft.half()
                    wimg = _warp_flow(buf[rel[i + 1]:rel[i + 1] + 1], ft, grid)
                    w = (temporal_strength * _flow_motion_weight(ft)).to(acc.dtype)
                    acc = acc + w * wimg
                    denom = denom + w
                fused[j:j + 1] = acc / denom.clamp_min(1e-4)
            chunk = fused
        if detail_amount > 0:
            chunk = _detail_enhance_gpu(chunk, detail_amount, detail_radius, detail_mode)
        out_chunks.append(chunk.float().permute(0, 2, 3, 1).cpu())
    return torch.cat(out_chunks, dim=0)


# ---------------------------------------------------------------------------
# Face restoration (small / distant broken faces)
# YOLOv8-Face detect -> GFPGAN / CodeFormer ONNX restore -> seamless blend back.
# Zero new pip deps: uses bundled ultralytics + onnxruntime + torch's cudnn.
# ---------------------------------------------------------------------------
_FACE_CACHE: dict = {}
_ORT_DLL_READY: list = [False]


def _ensure_ort_dll() -> None:
    """onnxruntime CUDA EP needs cudnn/cublas DLLs; they ship inside torch/lib."""
    if _ORT_DLL_READY[0]:
        return
    try:
        import site
        for p in site.getsitepackages():
            d = os.path.join(p, "torch", "lib")
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "cudnn64_9.dll")):
                os.add_dll_directory(d)
                break
    except Exception:
        pass
    _ORT_DLL_READY[0] = True


def _face_models(mode: str):
    """Lazy-load (and cache) detector + restorer for a face mode."""
    key = mode
    if key in _FACE_CACHE:
        return _FACE_CACHE[key]
    _ensure_ort_dll()
    import onnxruntime as ort
    from ultralytics import YOLO
    base = getattr(folder_paths, "models_dir", "models")
    det_path = os.path.join(base, "ultralytics", "bbox", "face_yolov8m.pt")
    if not os.path.exists(det_path):
        for cand in ("face_yolov8s.pt", "face_yolov8n.pt"):
            p2 = os.path.join(base, "ultralytics", "bbox", cand)
            if os.path.exists(p2):
                det_path = p2
                break
    if mode == "CodeFormer":
        onnx_path = os.path.join(base, "insightface", "codeformer.onnx")
        if not os.path.exists(onnx_path):
            onnx_path = os.path.join(base, "facerestore_models", "codeformer.onnx")
    else:
        onnx_path = os.path.join(base, "facerestore_models", "GFPGANv1.4.onnx")
        if not os.path.exists(onnx_path):
            onnx_path = os.path.join(base, "facerestore_models", "GFPGANv1.3.onnx")
    if not os.path.exists(det_path):
        raise FileNotFoundError("face detector not found (ultralytics/bbox/face_yolov8*.pt)")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"face restore model not found: {onnx_path}")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    det = YOLO(det_path)
    _FACE_CACHE[key] = (det, sess)
    return det, sess


def _restore_faces_frame(img, boxes, sess, mode, blend, fidelity=0.75):
    """Restore every detected face in one BGR/RGB uint8 frame (numpy), blend back.

    v1.7.0 upgrades (对标 Topaz / FlashVSR 的自然脸):
      (1) fidelity-first defaults (blend 0.65 / fidelity 0.75): keep original
          structure, only gently regenerate — no more hallucinated features;
      (2) adaptive strength: the smaller / more distant the face, the LOWER the
          replacement strength and the HIGHER the fidelity — tiny faces are
          barely touched instead of being redrawn into broken/misshapen ones;
      (3) input enhancement: LANCZOS upsample + horizontal-flip TTA averaging
          to stabilise reconstruction and suppress asymmetry / artifacts.
    Also keeps v1.4.2 fixes: aspect-preserving letterbox into 512x512 (no
    anisotropic stretch), elliptical blend mask centred on the face bbox, and
    generous vertical padding to keep chin/forehead inside.
    """
    H, W = img.shape[:2]
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b]
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 8:
            continue
        # ---- (2) adaptive strength from face size (relative to frame) ----
        ratio = (w * h) / float(max(H * W, 1))
        if ratio >= 0.02:
            strength = 1.0
        else:
            strength = max(0.35, ratio / 0.02)
        blend_eff = blend * strength
        fid_eff = fidelity
        if mode == "CodeFormer":
            # small/distant faces: push toward structure-preserving (less regen)
            fid_eff = min(1.0, fidelity + (1.0 - strength) * 0.30)
        if strength < 0.5:
            # very small faces: barely touch (avoid hallucinating features)
            blend_eff *= 0.6
        # generous pad (more vertical) so the whole face stays inside the crop
        pad_w, pad_h = 0.45 * w, 0.50 * h
        cx1 = max(0, int(x1 - pad_w)); cy1 = max(0, int(y1 - pad_h))
        cx2 = min(W, int(x2 + pad_w)); cy2 = min(H, int(y2 + pad_h))
        crop = img[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        # --- aspect-preserving letterbox into 512x512 (no anisotropic stretch) ---
        S = 512
        scale = min(S / float(ch), S / float(cw))
        nh = max(1, int(round(ch * scale))); nw = max(1, int(round(cw * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.zeros((S, S, 3), dtype=np.uint8)
        dx, dy = (S - nw) // 2, (S - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        inp = canvas.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]

        def _run(t):
            feed = {sess.get_inputs()[0].name: t}
            if len(sess.get_inputs()) > 1:  # CodeFormer: weight (fidelity)
                feed[sess.get_inputs()[1].name] = np.array([float(fid_eff)], dtype=np.float64)
            return sess.run(None, feed)[0][0].transpose(1, 2, 0)

        # --- (3) flip-TTA averaging to stabilise reconstruction ----
        o = np.clip(_run(inp), 0, 1)
        if strength >= 0.5:
            of = np.clip(_run(inp[:, :, :, ::-1].copy()), 0, 1)
            o = 0.5 * (o + of[:, :, ::-1])
        o = (o * 255.0).astype(np.uint8)
        restored = o[dy:dy + nh, dx:dx + nw]            # drop letterbox padding
        restored = cv2.resize(restored, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        # --- elliptical mask centred on the FACE bbox (not crop centre) ---
        fx = (x1 + x2) / 2.0 - cx1
        fy = (y1 + y2) / 2.0 - cy1
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        sx = (xx - fx) / max(w / 2.0 + 0.30 * cw, 1.0)
        sy = (yy - fy) / max(h / 2.0 + 0.35 * ch, 1.0)
        m = np.clip(1.0 - np.sqrt(sx * sx + sy * sy), 0.0, 1.0)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(2.0, min(cw, ch) / 28.0))
        m = m[..., None].astype(np.float32)
        fused = blend_eff * restored.astype(np.float32) + (1.0 - blend_eff) * crop.astype(np.float32)
        out[cy1:cy2, cx1:cx2] = (m * fused + (1.0 - m) * crop.astype(np.float32)).astype(np.uint8)
    return out


def _face_restore_frames(out_tensor, mode, det_conf, blend, fidelity=0.75):
    """Apply face restoration to an SR tensor [n,H,W,3] float 0-1 (CPU).

    Returns (out_tensor, total_faces_detected)."""
    if mode == "Off" or not _HAS_CV2 or not torch.cuda.is_available():
        return out_tensor, 0
    try:
        det, sess = _face_models(mode)
    except Exception as e:  # never crash the main upscale on face issues
        print(f"[BSAI-H3] face restore unavailable ({e}); skipping")
        return out_tensor, 0
    n = out_tensor.shape[0]
    det_bs = min(8, n)
    chunks = []
    total = 0
    for s in range(0, n, det_bs):
        seg = out_tensor[s:s + det_bs]
        frames = (seg.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        # pass as a list so ultralytics letterboxes each frame independently
        res = det.predict([frames[i] for i in range(len(frames))],
                          conf=det_conf, imgsz=1280, verbose=False)
        for i in range(len(frames)):
            boxes = res[i].boxes.xyxy.cpu().numpy() if (res[i].boxes is not None) else np.zeros((0, 4))
            if len(boxes):
                total += len(boxes)
                frames[i] = _restore_faces_frame(frames[i], boxes, sess, mode, blend, fidelity)
        chunks.append(torch.from_numpy(frames).float() / 255.0)
    return torch.cat(chunks, dim=0), total


# ---------------------------------------------------------------------------
# Unified third-party engine wrappers (FlashVSR / SeedVR2 / NVIDIA RTX)
# Each keeps its own model weights path unchanged; loaded lazily so the
# plugin still imports fine if any engine is missing.
# ---------------------------------------------------------------------------
def _load_plugin_package(plugin_dir, pkg_name):
    """Load a ComfyUI custom_nodes plugin as a package (supports relative imports).
    No explicit 'plugin not installed' error — if the plugin is missing the subsequent
    import raises naturally. We only proactively error on missing model weights."""
    import importlib.util
    if pkg_name not in sys.modules:
        init_path = os.path.join(plugin_dir, "__init__.py")
        spec = importlib.util.spec_from_file_location(
            pkg_name, init_path if os.path.exists(init_path) else os.path.join(plugin_dir, "nodes.py"),
            submodule_search_locations=[plugin_dir])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = pkg
        spec.loader.exec_module(pkg)
    return sys.modules[pkg_name]


def _flashvsr_upscale(frames, scale, seed=42):
    """FlashVSR-v1.1 diffusion video SR. Weights: ComfyUI/models/FlashVSR-v1.1/ (unchanged)."""
    import importlib
    model_dir = os.path.join(folder_paths.models_dir, "FlashVSR-v1.1")
    required = ["diffusion_pytorch_model_streaming_dmd.safetensors", "Wan2.1_VAE.pth",
                "LQ_proj_in.ckpt", "TCDecoder.ckpt"]
    missing = [f for f in required if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        raise RuntimeError(
            "FlashVSR-v1.1 模型权重缺失 / FlashVSR-v1.1 weights missing:\n"
            f"  缺失 / Missing: {', '.join(missing)}\n"
            "  下载 / Download: https://huggingface.co/JunhaoZhuang/FlashVSR\n"
            f"  放置路径 / Place in: {model_dir}\n"
            f"  需要文件 / Required: {', '.join(required)}"
        )
    flash_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-FlashVSR_Ultra_Fast")
    _load_plugin_package(flash_dir, "_bsai_flashvsr")
    nodes_mod = importlib.import_module("_bsai_flashvsr.nodes")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = nodes_mod.init_pipeline("FlashVSR-v1.1", "tiny", dev, torch.bfloat16)
    s = max(2, min(4, int(round(float(scale)))))
    out = nodes_mod.flashvsr(pipe, frames, s, True, True, True, 256, 24, False, 2.0, 3.0, 11, seed, True)
    if out.is_cuda:
        out = out.cpu()
    return out.float().clamp(0, 1)


def _seedvr2_upscale(frames, scale, seed=42):
    """SeedVR2 7B diffusion video SR. Weights: ComfyUI/models/SEEDVR2/ (unchanged)."""
    import importlib
    model_dir = os.path.join(folder_paths.models_dir, "SEEDVR2")
    dit_file = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    vae_file = "ema_vae_fp16.safetensors"
    missing = [f for f in [dit_file, vae_file] if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        raise RuntimeError(
            "SeedVR2 模型权重缺失 / SeedVR2 weights missing:\n"
            f"  缺失 / Missing: {', '.join(missing)}\n"
            "  下载 / Download: https://huggingface.co/ByteDance/SeedVR2-7B (或社区镜像)\n"
            f"  放置路径 / Place in: {model_dir}\n"
            f"  需要文件 / Required: {dit_file}, {vae_file}"
        )
    seed_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-SeedVR2_VideoUpscaler")
    _load_plugin_package(seed_dir, "_bsai_seedvr2")
    up_mod = importlib.import_module("_bsai_seedvr2.src.interfaces.video_upscaler")
    SeedVR2VideoUpscaler = up_mod.SeedVR2VideoUpscaler
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dit = {
        "model": dit_file,
        "device": dev, "offload_device": "none", "cache_model": False,
        "blocks_to_swap": 0, "swap_io_components": False, "attention_mode": "sdpa",
        "torch_compile_args": None, "node_id": "bsai_unified",
    }
    vae = {
        "model": vae_file,
        "device": dev, "offload_device": "none", "cache_model": False,
        "encode_tiled": False, "encode_tile_size": 512, "encode_tile_overlap": 64,
        "decode_tiled": False, "decode_tile_size": 512, "decode_tile_overlap": 64,
        "tile_debug": False, "torch_compile_args": None, "node_id": "bsai_unified",
    }
    h, w = frames.shape[1], frames.shape[2]
    target_res = int(round(min(h, w) * float(scale)))
    target_res = max(16, target_res - target_res % 2)
    result = SeedVR2VideoUpscaler.execute(
        image=frames, dit=dit, vae=vae, seed=seed,
        resolution=target_res, max_resolution=0, batch_size=5,
        uniform_batch_size=False, temporal_overlap=0, prepend_frames=0,
        color_correction="lab", input_noise_scale=0.0, latent_noise_scale=0.0,
        offload_device="cpu", enable_debug=False,
    )
    out = result[0] if isinstance(result, (tuple, list)) else result
    if hasattr(out, "result"):
        out = out.result
    if torch.is_tensor(out):
        if out.is_cuda:
            out = out.cpu()
        if out.dtype != torch.float32:
            out = out.float()
        return out.clamp(0, 1)
    return frames


def _rtx_upscale(frames, scale, quality="超高"):
    """NVIDIA RTX Video Super Resolution (nvidia-vfx, no model files)."""
    import importlib
    yuan_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-Yuan-Tool")
    _load_plugin_package(yuan_dir, "_bsai_yuan")
    rtx_mod = importlib.import_module("_bsai_yuan.Yuan_RTX_Upscale")
    YuanRTXVideoUpscaleH3 = rtx_mod.YuanRTXVideoUpscaleH3
    resize_params = {"resize_type": "按倍数缩放", "scale": float(scale), "width": 0, "height": 0}
    out = YuanRTXVideoUpscaleH3._run_super_resolution(frames, resize_params, quality)
    if out.is_cuda:
        out = out.cpu()
    return out.float().clamp(0, 1)


# ---------------------------------------------------------------------------
# Main node: video frames -> AI upscaled frames
# ---------------------------------------------------------------------------
class BSAI_H3_Upscale4K:
    """Video frame AI super-resolution for MiniMax H3 (pixel domain, extremely fast)."""

    # Third-party diffusion / GPU engine options (prepended to the model list).
    # Each keeps its own weights path unchanged; loaded lazily via wrappers above.
    ENGINE_OPTIONS = {
        "FlashVSR-v1.1 (扩散视频超分)": "flashvsr",
        "SeedVR2 7B (扩散视频超分)": "seedvr2",
        "NVIDIA RTX Video Super Res": "rtx",
    }

    # Generative Topaz engine options (prepended to the Real-ESRGAN model list).
    # When selected, the node routes to _topaz_upscale (neuroserver) instead of
    # Real-ESRGAN — single-node "完美档" with our detail/softness/face-restore
    # post-processing still available on top.
    TOPAZ_OPTIONS = {
        "Topaz 星光 2.6 (生成式完美档)": "slp-26",
        "Topaz Astra (生成式)": "astra",
    }

    @classmethod
    def INPUT_TYPES(cls):
        models = list(cls.ENGINE_OPTIONS.keys()) + list(cls.TOPAZ_OPTIONS.keys()) + list_available_models()
        return {
            "required": {
                "images / 图像": ("IMAGE",),
                # general-x4v3 default: ~8x faster than x4plus with nearly identical
                # quality (PSNR ~39 dB on test frames). x4plus = max quality, slowest.
                "model_name / 模型": (models, {"default": "realesr-general-x4v3.pth"}),
                # Scale supports any value 1.0-8.0 (incl. Topaz-style precise ratios
                # like 2.67 / 3.0): integer multiples of the model run directly,
                # other ratios are super-resolved to the next model integer scale and
                # then resized to the exact target (even-pixel aligned). 4 = 4K class.
                "scale / 放大倍数": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.01}),
                # tile_size=0 => full-image fast path (recommended on RTX 30xx+).
                # Use a positive tile only if you run out of VRAM.
                "tile_size / 分块大小": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16}),
                "tile_pad / 分块重叠": ("INT", {"default": 16, "min": 0, "max": 128, "step": 4}),
                "batch_frames / 批帧数": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "use_fp16 / 半精度": ("BOOLEAN", {"default": True}),
                # torch.compile: ~1.6-1.9x faster on fixed-size video frames.
                # One-time compile cost on first run, then cached process-wide.
                "use_compile / 编译加速": ("BOOLEAN", {"default": True}),
                # Temporal consistency: motion-compensated blend with neighbours
                # (Farneback optical flow on LR, GPU warp on SR). 0 = off.
                "temporal_strength / 时序强度": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.8, "step": 0.05}),
                # Detail enhancement: separable-Gaussian unsharp mask on SR. 0 = off.
                "detail_amount / 细节强度": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.5, "step": 0.05}),
                "detail_radius / 细节半径": ("FLOAT", {"default": 1.8, "min": 0.3, "max": 8.0, "step": 0.1}),
                # Softness (borrowed from Topaz Starlight): after detail USM, blend
                # back a small fraction of a Gaussian-blurred copy to tame overshoot
                # and blocky artifacts. 0 = off, ~0.3 gentle, 1.0 very soft.
                "softness / 柔和度": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Face restoration: detects faces (YOLOv8-Face) then regenerates
                # facial structure with GFPGAN / CodeFormer (ONNX, GPU, zero new
                # pip deps). Fixes H3's small / distant broken faces.
                "face_restore / 人脸修复": (["Off", "GFPGANv1.4", "CodeFormer"], {"default": "Off"}),
                # Face detector confidence threshold (lower = more detections,
                # including tiny distant faces; may add false positives).
                "face_det_conf / 检测置信度": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95, "step": 0.05}),
                # How strongly the restored face is blended over the original
                # crop. 1.0 = full restore, ~0.65 = fidelity-first (keep skin
                # and original structure; smaller faces are blended even less).
                "face_blend / 融合强度": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.0, "step": 0.05}),
                # CodeFormer fidelity (0 = heavy regeneration for badly-broken
                # faces, 1 = preserve original structure/details). Ignored by
                # GFPGANv1.4. Default 0.75 = fidelity-first: avoids the AI
                # 'hallucinated' distorted features seen on small/distant faces.
                "face_fidelity / 保真度": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Detail rebuild mode: classic = unsharp mask only; smart =
                # unsharp + light local-contrast rebuild (generative-style
                # texture, gated to avoid halos) — reads closer to Topaz /
                # FlashVSR's reconstructed texture.
                "detail_mode / 细节模式": (["classic", "smart"], {"default": "smart"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE / 图像", "width / 宽", "height / 高", "scale_used / 实际倍率", "info / 信息")
    FUNCTION = "upscale"
    CATEGORY = "BSAI/H3"
    DESCRIPTION = (
        "H3 视频专用 AI 超分（像素域）：多引擎集合节点\n"
        "  • Real-ESRGAN 极速档：general-x4v3 / x4plus，光流时序 + 多尺度细节 + 人脸修复\n"
        "  • FlashVSR-v1.1 / SeedVR2 7B：扩散式视频超分（各自权重路径不变）\n"
        "  • NVIDIA RTX Video Super Res：nvidia-vfx GPU 超分（无模型文件）\n"
        "  • Topaz 生成式完美档：星光 2.6 / Astra 神经引擎（本机 ComfyUI/models/Topaz_Engine），\n"
        "    单节点即可达 Topaz 官方级人脸细节与纹理，后续 detail/softness/face_restore 仍可叠加。\n"
        "Unified multi-engine video super-resolution node: Real-ESRGAN fast tier, "
        "FlashVSR / SeedVR2 diffusion tiers (own weight paths), NVIDIA RTX VSR, "
        "and Topaz generative tier (Starlight 2.6 / Astra), all with our post-processing on top."
    )
    def upscale(self, **kw):
        g = kw.get
        images = g("images / 图像")
        model_name = g("model_name / 模型")
        scale = g("scale / 放大倍数", 4.0)
        tile_size = g("tile_size / 分块大小", 0)
        tile_pad = g("tile_pad / 分块重叠", 16)
        batch_frames = g("batch_frames / 批帧数", 4)
        use_fp16 = g("use_fp16 / 半精度", True)
        use_compile = g("use_compile / 编译加速", True)
        temporal_strength = g("temporal_strength / 时序强度", 0.2)
        detail_amount = g("detail_amount / 细节强度", 0.5)
        detail_radius = g("detail_radius / 细节半径", 1.8)
        softness = g("softness / 柔和度", 0.1)
        face_restore = g("face_restore / 人脸修复", 'Off')
        face_det_conf = g("face_det_conf / 检测置信度", 0.25)
        face_blend = g("face_blend / 融合强度", 0.65)
        face_fidelity = g("face_fidelity / 保真度", 0.75)
        detail_mode = g("detail_mode / 细节模式", 'smart')
        t0 = time.time()
        _engine = self.ENGINE_OPTIONS.get(model_name)
        _is_topaz = model_name in self.TOPAZ_OPTIONS

        if _engine == "flashvsr":
            # --- FlashVSR-v1.1 diffusion path -------------------------------
            out = _flashvsr_upscale(images, scale)
            eff_scale = float(out.shape[1] / float(images.shape[1]))
            temporal_strength = 0.0  # engine handles temporal consistency
            lr_np = None
        elif _engine == "seedvr2":
            # --- SeedVR2 7B diffusion path ----------------------------------
            out = _seedvr2_upscale(images, scale)
            eff_scale = float(out.shape[1] / float(images.shape[1]))
            temporal_strength = 0.0
            lr_np = None
        elif _engine == "rtx":
            # --- NVIDIA RTX Video Super Resolution path ---------------------
            out = _rtx_upscale(images, scale)
            eff_scale = float(out.shape[1] / float(images.shape[1]))
            temporal_strength = 0.0
            lr_np = None
        elif _is_topaz:
            # --- Topaz generative engine path (生成式完美档) -------------------
            # Routes to _topaz_upscale (neuroserver, Starlight 2.6 / Astra).
            # Topaz engine already does temporal consistency, so our optical-flow
            # temporal pass is skipped; detail/softness/face-restore still apply.
            model_id = self.TOPAZ_OPTIONS[model_name]
            topaz_scale = min(float(scale), 4.0)  # engine supports up to 4x
            frames = (images * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
            out_np = _topaz_upscale(frames, fps=24, scale=topaz_scale,
                                     strength=1.0, max_gpu_mem=14.0, qp=14,
                                     model_id=model_id)
            out = torch.from_numpy(out_np.astype(np.float32) / 255.0)
            eff_scale = float(out.shape[1] / float(images.shape[1]))
            temporal_strength = 0.0  # engine handles temporal consistency
            lr_np = None
        else:
            # --- Real-ESRGAN classic path ---------------------------------------
            path = ensure_model(model_name)
            if use_compile and torch.cuda.is_available():
                model = _load_model_compiled(path, use_fp16)
            else:
                model = _load_model(path, use_fp16)
            scale = float(scale)
            ms = int(model.scale)

            # keep the original LR frames around (needed by the temporal pass for flow)
            lr_np = None
            if temporal_strength > 0 and _HAS_CV2 and images.shape[0] > 1:
                lr_np = np.ascontiguousarray(images.float().numpy(), dtype=np.float32)

            # --- scale plan (Topaz-style arbitrary ratios) -------------------------
            # Super-resolve to the smallest model-integer power >= requested scale,
            # then (only if the ratio is not an exact model multiple) precisely resize
            # to the target with even-pixel alignment.
            n = 0
            sr = 1.0
            while sr < scale - 1e-6:
                sr *= ms
                n += 1
            out = images
            for _ in range(max(1, n)):
                out = _upscale_batch(model, out, tile_size, tile_pad, batch_frames)
            if abs(sr - scale) > 1e-3:
                in_h, in_w = images.shape[1], images.shape[2]
                th = int(round(in_h * scale)); tw = int(round(in_w * scale))
                th += th % 2; tw += tw % 2
                out = F.interpolate(out.permute(0, 3, 1, 2), size=(th, tw),
                                    mode="bilinear", align_corners=False)
                out = out.permute(0, 2, 3, 1).contiguous()
            eff_scale = float(out.shape[1] / float(images.shape[1]))

        # Temporal consistency (motion-compensated neighbour blend) + detail USM
        t_td = time.time()
        out = _video_temporal_detail(out, lr_np, temporal_strength, detail_amount,
                                     detail_radius, eff_scale, detail_mode)
        td_elapsed = time.time() - t_td

        # Softness (Topaz-style) — GPU, then back to CPU
        if softness > 0 and torch.cuda.is_available():
            dev = torch.cuda.current_device()
            out = _soften_gpu(out.to(dev), softness).cpu()

        # Face restoration (small / distant broken faces) - optional, on the SR frames
        t_fr = time.time()
        if face_restore != "Off":
            out, _ = _face_restore_frames(out, face_restore, face_det_conf, face_blend, face_fidelity)
        fr_elapsed = time.time() - t_fr

        bh, bw = out.shape[1], out.shape[2]
        elapsed = time.time() - t0
        info = (
            f"model: {model_name} (eff_scale={eff_scale:.2f}x) | "
            f"output: {bw}x{bh} | "
            f"fp16: {use_fp16} | compile: {use_compile} | tile: {tile_size} pad:{tile_pad} | "
            f"temporal: {temporal_strength} | detail: {detail_amount}@{detail_radius} "
            f"({detail_mode}) | "
            f"softness: {softness} | "
            f"face: {face_restore} (conf={face_det_conf}, blend={face_blend}, fid={face_fidelity}) | "
            f"frames: {images.shape[0]} | time: {elapsed:.2f}s "
            f"(temporal+detail: {td_elapsed:.2f}s, face: {fr_elapsed:.2f}s) | "
            f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
        )
        return (out, bw, bh, float(eff_scale), info)


# ===========================================================================
# Learned H3 latent upscaler (3D conv network)
# ---------------------------------------------------------------------------
# 方法借鉴自 Comfyui_Minimax_h3_latent_Upscaler (LBH-123-AI)：用训练好的 3D
# 卷积网络在 24 通道 H3 VAE latent 空间放大（比双线性/双三次插值更干净，可配合
# H3 第二遍采样精修）。仅借鉴方法，不包含任何第三方权重；模型文件由用户放入
# ComfyUI/models/latent_upscale_models/（如 minimax_h3_latent_upscaler_3d_fp16.safetensors）。
# 实现为 torch 原生（零 einops 依赖），推理时自动检测网络结构并加载权重。
# ===========================================================================
import glob
import re

_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER),
    )

# 24-channel Minimax H3 latent normalization stats (from the upscaler's training code).
_LATENTS_MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
                 -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
                 -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
                 -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
                 -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
                 -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
_LATENTS_STD = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
                1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
                0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
                0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
                2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
                3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]


def _latent_norm_tensors(device, dtype):
    mean = torch.tensor(_LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(_LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _gn3d(channels):
    return nn.GroupNorm(32, channels)


def _zero_mod3d(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class _Attn3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = _gn3d(c)
        self.q = nn.Conv3d(c, c, 1)
        self.k = nn.Conv3d(c, c, 1)
        self.v = nn.Conv3d(c, c, 1)
        self.proj_out = nn.Conv3d(c, c, 1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).flatten(2).transpose(1, 2).unsqueeze(1)  # (B,1,L,C)
        k = self.k(h).flatten(2).transpose(1, 2).unsqueeze(1)
        v = self.v(h).flatten(2).transpose(1, 2).unsqueeze(1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1).transpose(1, 2).view(B, C, T, H, W)
        return x + self.proj_out(h)


class _ResEmb3D(nn.Module):
    def __init__(self, c, emb_c, dropout=0, out_c=None):
        super().__init__()
        self.out_channels = out_c or c
        self.in_layers = nn.Sequential(_gn3d(c), nn.SiLU(), nn.Conv3d(c, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_c, 2 * self.out_channels))
        self.out_norm = _gn3d(self.out_channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=dropout),
                                        _zero_mod3d(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)))
        self.skip = nn.Conv3d(c, self.out_channels, 1) if self.out_channels != c else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class _TemporalConv3D(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        p = k // 2
        self.norm = _gn3d(c)
        self.dwconv = nn.Conv3d(c, c, kernel_size=(k, 1, 1), padding=(p, 0, 0), groups=c)
        self.pwconv = nn.Conv3d(c, c, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        return x + self.pwconv(F.silu(self.dwconv(self.norm(x))))


class _LatentResizer3D(nn.Module):
    """Pure-3D latent resizer (arch identical to the reference upscaler)."""

    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512,
                 dropout=0.1, attn=False, temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(_Attn3D(channels))
            self.in_blocks.append(_ResEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(_TemporalConv3D(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(_Attn3D(channels))
            self.out_blocks.append(_ResEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(_TemporalConv3D(channels, temporal_kernel))
        self.norm_out = _gn3d(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x
        if size == x.shape[-3:]:
            return x
        emb = self.embed(torch.tensor([scale - 1 if scale is not None else 0.0],
                                      dtype=x.dtype, device=x.device).unsqueeze(0))
        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, _ResEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            if isinstance(b, _ResEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)
        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


_LATENT_MODEL_CACHE = {}


def _scan_latent_models():
    try:
        dirs = folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)
    except Exception:
        dirs = [os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)]
    out = []
    for d in dirs:
        for ext in ("*.safetensors", "*.pth"):
            out.extend(glob.glob(os.path.join(d, ext)))
    return sorted(os.path.basename(f) for f in out)


def _pick_default_latent_model(models):
    for pref in ("minimax_h3_latent_upscaler_3d", "minimax_h3", "minimax"):
        for m in models:
            if pref in m:
                return m
    return models[0]


def _load_latent_3d_model(name, device, precision):
    key = f"{name}::{device}::{precision}"
    if key in _LATENT_MODEL_CACHE:
        return _LATENT_MODEL_CACHE[key]
    try:
        dirs = folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)
    except Exception:
        dirs = [os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)]
    path = next((os.path.join(d, name) for d in dirs if os.path.exists(os.path.join(d, name))), None)
    if path is None:
        raise FileNotFoundError(f"latent upscaler model not found: {name} in {dirs}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k: (v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v) for k, v in sd.items()}
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    cfg = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
           "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5}
    if "conv_in.weight" in sd:
        cfg["in_channels"] = sd["conv_in.weight"].shape[1]
        cfg["channels"] = sd["conv_in.weight"].shape[0]
    in_ids, out_ids, t_in, t_out = set(), set(), set(), set()
    for k in sd.keys():
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", k)
        if m:
            out_ids.add(int(m.group(1)))
        m = re.match(r"in_blocks\.(\d+)\.dwconv\.weight", k)
        if m:
            t_in.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.dwconv\.weight", k)
        if m:
            t_out.add(int(m.group(1)))
    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    if t_in or t_out:
        cfg["temporal_every"] = 2
        for k in sd:
            if k.endswith("dwconv.weight"):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0
    model = _LatentResizer3D(**cfg)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float16)
    if dt != torch.float32:
        model = model.to(dt)
    _LATENT_MODEL_CACHE[key] = model
    print(f"[BSAI-H3] loaded learned latent upscaler: {name} "
          f"({cfg['in_blocks']}in/{cfg['out_blocks']}out, ch={cfg['channels']}, "
          f"temporal={'on' if cfg['temporal_every'] else 'off'}) {precision}")
    return model


# ---------------------------------------------------------------------------
# Latent node: H3 latent -> enlarged latent (32px aligned, second-pass refine)
#   - learned-3d (default): trained 3D-conv latent upscaler (borrowed method)
#   - nearest-exact / bilinear / area / bicubic / bislerp: classic interpolation
# ---------------------------------------------------------------------------
class BSAI_H3_Upscale4K_Latent:
    """H3 latent-space upscale aligned to the 32-px grid for safe second-pass sampling."""

    upscale_methods = ["learned-3d", "nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(cls):
        models = _scan_latent_models()
        if models:
            model_opts = models
            model_def = _pick_default_latent_model(models)
        else:
            model_opts = ["(未找到模型，请放入 models/latent_upscale_models/)"]
            model_def = model_opts[0]
        return {
            "required": {
                "samples / 潜空间": ("LATENT",),
                "upscale_method / 放大方法": (cls.upscale_methods, {"default": "learned-3d"}),
                "scale_by / 放大倍数": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.01}),
                "model_name / 模型": (model_opts, {"default": model_def}),
                "precision / 精度": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("LATENT / 潜空间", "width / 宽", "height / 高", "effective_scale / 实际倍率", "info / 信息")
    FUNCTION = "upscale"
    CATEGORY = "BSAI/H3"
    DESCRIPTION = (
        "H3 专属 latent 放大：自动 32 像素对齐，供 H3 第二遍采样补细节（保持人物一致性）。\n"
        "learned-3d = 训练好的 3D 卷积 latent 超分网络（方法借鉴 Minimax_h3_latent_Upscaler，\n"
        "需在 models/latent_upscale_models/ 放置 minimax_h3_latent_upscaler_3d_fp16.safetensors）；\n"
        "其余为经典插值回退。\n"
        "H3-specific latent upscale with 32-px alignment: learned-3d = trained 3D-conv\n"
        "latent SR network (method borrowed, weight in models/latent_upscale_models/),\n"
        "other methods = classic interpolation fallback."
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

    def _upscale_learned(self, samples, source, lw, lh, scale_by, model_name, precision):
        if scale_by < 1.0:
            raise ValueError("learned-3d 仅支持放大 (scale_by >= 1.0)")
        if scale_by > 4.0:
            raise ValueError("learned-3d 网络仅支持 1.0-4.0 倍放大，请调低 scale_by")
        if not torch.cuda.is_available():
            raise RuntimeError("learned-3d 需要 CUDA GPU")
        models = _scan_latent_models()
        if not models:
            raise FileNotFoundError(
                "未找到学习型 latent 模型。请将 minimax_h3_latent_upscaler_3d_fp16.safetensors "
                "放入 ComfyUI/models/latent_upscale_models/")
        name = model_name if (model_name in models) else _pick_default_latent_model(models)
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        dev = torch.device("cuda")
        model = _load_latent_3d_model(name, dev, precision)
        s = source.to(device=dev, dtype=dt, copy=True)
        was_4d = s.dim() == 4
        if was_4d:
            s = s.unsqueeze(2)
        b, c, t, h_in, w_in = s.shape
        ds = self.H3_VAE_SPATIAL_DOWNSCALE  # 16
        # pixel-space target (scale by multiplier)
        w_px = w_in * ds * scale_by
        h_px = h_in * ds * scale_by
        # align to 32-px pixel grid, width drives, keep proportion
        align = 32
        w_px_a = round(w_px / align) * align
        h_px_a = w_px_a / (w_in / h_in)
        # snap to VAE grid so latent sizes are exact integers
        w_px_f = round(w_px_a / ds) * ds
        h_px_f = round(h_px_a / ds) * ds
        w_out = max(1, int(w_px_f // ds))
        h_out = max(1, int(h_px_f // ds))
        if w_out == w_in and h_out == h_in:
            return (samples, w_in * ds, h_in * ds, 1.0,
                    f"learned-3d: no-op (same size) | model={name} | {precision}")
        norm_mean, norm_std = _latent_norm_tensors(dev, dt)
        with torch.inference_mode():
            s.sub_(norm_mean).div_(norm_std)
            out = model(s, scale=scale_by, target_size=(t, h_out, w_out))
            del s
            out.mul_(norm_std).add_(norm_mean)
        if was_4d:
            out = out.squeeze(2)
        out = out.to(device="cpu", dtype=source.dtype)
        torch.cuda.empty_cache()
        result = samples.copy()
        result["samples"] = out
        eff_actual = w_out / w_in
        info = (f"latent {lw}x{lh} -> {w_out}x{h_out} | pixel {w_in * ds}x{h_in * ds} -> "
                f"{w_out * ds}x{h_out * ds} | eff_scale {eff_actual:.4f}x | "
                f"learned-3d: {name} | {precision}")
        return (result, w_out * ds, h_out * ds, float(eff_actual), info)
    def upscale(self, **kw):
        g = kw.get
        samples = g("samples / 潜空间")
        upscale_method = g("upscale_method / 放大方法", 'learned-3d')
        scale_by = g("scale_by / 放大倍数", 1.5)
        model_name = g("model_name / 模型")
        precision = g("precision / 精度", 'fp16')
        import comfy.utils
        source = samples["samples"]
        lw, lh = source.shape[-1], source.shape[-2]
        if upscale_method == "learned-3d":
            return self._upscale_learned(samples, source, lw, lh, scale_by, model_name, precision)
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


# ---------------------------------------------------------------------------
# Standalone face-restore node: works on any video / image frames, no upscale.
# Fixes H3 small / distant broken faces anywhere in a workflow.
# ---------------------------------------------------------------------------
class BSAI_H3_FaceRestore:
    """Detect faces (YOLOv8-Face) and regenerate facial structure with
    GFPGAN / CodeFormer (ONNX + GPU). Works standalone on any IMAGE frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images / 图像": ("IMAGE",),
                "face_restore / 人脸修复": (["Off", "GFPGANv1.4", "CodeFormer"], {"default": "GFPGANv1.4"}),
                "face_det_conf / 检测置信度": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95, "step": 0.05}),
                "face_blend / 融合强度": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.0, "step": 0.05}),
                "face_fidelity / 保真度": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("IMAGE / 图像", "faces_detected / 检测人脸数", "info / 信息")
    FUNCTION = "restore"
    CATEGORY = "BSAI/H3"
    DESCRIPTION = (
        "独立人脸修复节点：YOLOv8-Face 检测 + GFPGAN/CodeFormer 重建五官，\n"
        "解决 H3 中远景小脸崩坏 / 五官模糊丢失，可在任意工作流单独使用。\n"
        "v1.7.0: 保真优先（blend 0.65 / fidelity 0.75）+ 自适应强度（小脸几乎不动，\n"
        "避免 AI 幻觉出变形五官）+ 翻转 TTA 平均，对标 Topaz / FlashVSR 自然脸。\n"
        "Standalone face restoration for any frames: YOLOv8-Face detect +\n"
        "GFPGAN/CodeFormer regenerate, fixes small/distant broken faces.\n"
        "v1.7.0: fidelity-first + adaptive strength + flip-TTA for natural faces."
    )
    def restore(self, **kw):
        g = kw.get
        images = g("images / 图像")
        face_restore = g("face_restore / 人脸修复", 'Off')
        face_det_conf = g("face_det_conf / 检测置信度", 0.25)
        face_blend = g("face_blend / 融合强度", 0.65)
        face_fidelity = g("face_fidelity / 保真度", 0.75)
        t0 = time.time()
        out, nf = _face_restore_frames(images, face_restore, face_det_conf, face_blend, face_fidelity)
        info = (
            f"face restore: {face_restore} (conf={face_det_conf}, blend={face_blend}, fid={face_fidelity}) | "
            f"faces detected: {nf} | frames: {images.shape[0]} | "
            f"time: {time.time() - t0:.2f}s | "
            f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
        )
        return (out, nf, info)


# ===========================================================================
# Topaz Starlight engine backend ("完美档" 底座)
# ---------------------------------------------------------------------------
# 底座 = Topaz 星光 2.6 神经引擎（用户本机 ComfyUI/topaz_engine，生成式扩散
# 重绘放大，效果对标官方）。本插件在其基础上继续叠加优化：
#   - 人脸修复（YOLO + GFPGAN/CodeFormer 保真模式，v1.7.0 自适应强度）
#   - 细节增强 / softness（可选）
# 引擎为商业版权：仅在本机调用用户已有引擎，不随插件分发任何引擎/权重文件。
# ===========================================================================
import subprocess
import tempfile
import uuid


def _topaz_engine_dir():
    """Topaz 引擎目录：默认 <ComfyUI>/models/Topaz_Engine，兼容旧 <ComfyUI>/topaz_engine。"""
    here = os.path.dirname(os.path.abspath(__file__))
    comfy = os.path.dirname(os.path.dirname(here))
    for cand in (os.path.join(comfy, "models", "Topaz_Engine"),
                 os.path.join(comfy, "topaz_engine")):
        if os.path.exists(cand):
            return cand
    return os.path.join(comfy, "models", "Topaz_Engine")


def _topaz_ffmpeg():
    eng = _topaz_engine_dir()
    for sub in ("bin", "bin171"):
        p = os.path.join(eng, sub, "ffmpeg.exe")
        if os.path.exists(p):
            return p, os.path.join(eng, sub, "ffprobe.exe")
    raise RuntimeError("Topaz 引擎 ffmpeg 未找到 - 请把引擎包放到 ComfyUI/models/Topaz_Engine (或旧路径 ComfyUI/topaz_engine)")


def _topaz_write_video(frames, fps, path, qp=14):
    """frames (B,H,W,3) uint8 RGB -> h264 nvenc mp4 (lossy intermediate)."""
    ffmpeg, _ = _topaz_ffmpeg()
    b, h, w, c = frames.shape
    cmd = [ffmpeg, '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-c:v', 'h264_nvenc', '-preset', 'p7', '-tune', 'hq',
           '-rc', 'constqp', '-qp', str(qp), '-pix_fmt', 'yuv420p', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = p.communicate(input=frames.tobytes())
    if p.returncode != 0:
        raise RuntimeError(f'ffmpeg encode failed: {err.decode(errors="replace")[:300]}')


def _topaz_read_video(path):
    """mp4 -> (B,H,W,3) uint8 RGB."""
    ffmpeg, ffprobe = _topaz_ffmpeg()
    pr = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path],
                        capture_output=True, text=True)
    w, h = pr.stdout.strip().split(',')
    w, h = int(w), int(h)
    p = subprocess.run([ffmpeg, '-loglevel', 'error', '-i', path,
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f'ffmpeg decode failed: {p.stderr.decode(errors="replace")[:200]}')
    raw = np.frombuffer(p.stdout, dtype=np.uint8)
    n = raw.size // (w * h * 3)
    return raw[: n * w * h * 3].reshape(n, h, w, 3)


def _topaz_run(in_path, out_path, scale, frames, w, h, strength, max_gpu_mem, model_id="slp-26"):
    """Run Topaz neuroserver (Starlight / Astra) on a video file."""
    ASTRA_MODELS = ('astra', 'astrahq', 'astrasharp', 'astrafast')
    if model_id in ASTRA_MODELS and frames < 9:
        raise RuntimeError(
            f'Astra 系列模型需要至少 9 帧输入 (当前 {frames} 帧), '
            '两版引擎的短视频路径均存在缺陷。请用更长的视频。')
    eng = _topaz_engine_dir()
    ns = os.path.join(eng, "neuroserver171", "neuroserver.exe")
    if not os.path.exists(ns):
        raise RuntimeError(f"neuroserver.exe 未找到: {ns} - 请把 Topaz 引擎包放到 ComfyUI/models/Topaz_Engine")
    model_store = os.path.join(eng, "models")
    tvmd = os.path.join(eng, "tvmd")
    lic = os.path.join(tvmd, "VR.lic") if os.path.exists(os.path.join(tvmd, "VR.lic")) \
        else os.path.join(model_store, "VR.lic")
    env = os.environ.copy()
    ffmpeg, _ = _topaz_ffmpeg()
    env['PATH'] = os.path.dirname(ffmpeg) + os.pathsep + env.get('PATH', '')
    env['TOPAZ_MODEL_STORE'] = model_store
    env['TVAI_MODEL_DIR'] = tvmd
    env['TOPAZLABS_LICENSE'] = lic
    # 星光带 softness=1（官方默认），Astra 家族不带
    if model_id == 'slp-26':
        filters = '[{"model": "%s", "enhancement_strength": %s, "softness": 1}]' % (model_id, strength)
    else:
        filters = '[{"model": "%s", "enhancement_strength": %s}]' % (model_id, strength)
    ow = int(round(w * scale)); ow += ow % 2
    oh = int(round(h * scale)); oh += oh % 2
    NS_ENC = ('-c:v h264_nvenc -profile:v high -pix_fmt yuv420p -g 30 -preset p7 -tune hq '
              '-rc constqp -qp 18 -rc-lookahead 20 -spatial_aq 1 -aq-strength 15 -b:v 0 -bf 0')
    cmd = [ns, '--once', '--input-path', in_path, '--output-path', out_path,
           '--start-frame-idx', '0', '--end-frame-idx', str(frames),
           '--max-gpu-mem', str(max_gpu_mem), '--filters', filters,
           '--output-width', str(ow), '--output-height', str(oh),
           '--upscale-factor', str(scale), '--ffmpeg-encoding', NS_ENC]
    print(f"[BSAI-H3/Topaz] 星光引擎: {ns} | 模型: {model_id} | {frames}帧 {w}x{h} -> {ow}x{oh} (x{scale})")
    proc = subprocess.Popen(cmd, env=env, cwd=os.path.dirname(ns),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding='utf-8', errors='replace')
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print("  [Topaz] " + line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'neuroserver failed (exit {proc.returncode})')
    if not os.path.isfile(out_path):
        raise RuntimeError('neuroserver produced no output file')


def _topaz_upscale(frames, fps, scale, strength, max_gpu_mem, qp=14, model_id="slp-26"):
    """frames (B,H,W,3) uint8 RGB -> Topaz engine upscaled (B,H',W',3) uint8."""
    tag = uuid.uuid4().hex[:10]
    tmp = tempfile.gettempdir()
    in_v = os.path.join(tmp, f'topaz_in_{tag}.mp4')
    out_v = os.path.join(tmp, f'topaz_out_{tag}.mp4')
    b, h, w, c = frames.shape
    try:
        _topaz_write_video(frames, fps, in_v, qp)
        _topaz_run(in_v, out_v, scale, b, w, h, strength, max_gpu_mem, model_id)
        return _topaz_read_video(out_v)
    finally:
        for f in (in_v, out_v):
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass


class BSAI_TopazEngine_FaceRestore:
    """Topaz Engine (星光/Astra 神经引擎) 放大 + 人脸修复叠加。

    底座=Topaz 神经引擎（默认 ComfyUI/models/Topaz_Engine，兼容旧 topaz_engine）。
    可选模型：星光 2.6 / Astra 家族（自由选择）。在引擎输出上继续叠加本插件
    优化：人脸修复（保真模式）+ 细节增强，实现「在其基础上继续优化高清放大
    与修复脸部细节」。
    """

    TOPAZ_MODELS = ["星光 2.6", "Astra", "Astra HQ", "Astra Sharp", "Astra Fast"]
    TOPAZ_MODEL_IDS = {"星光 2.6": "slp-26", "Astra": "astra", "Astra HQ": "astrahq",
                       "Astra Sharp": "astrasharp", "Astra Fast": "astrafast"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images / 图像": ("IMAGE",),
                "model / 模型": (cls.TOPAZ_MODELS, {"default": "星光 2.6",
                                    "tooltip": "星光2.6=默认最佳；Astra系列需≥9帧 / Starlight2.6 default, Astra needs >=9 frames"}),
                "scale / 放大倍数": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.01,
                                              "tooltip": "输出放大倍数 1-4（支持小数）/ Upscale factor 1-4 (fraction ok)"}),
                "enhancement_strength / 增强强度": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1,
                                                              "tooltip": "0.7柔和 / 1.0默认 / 1.3细节最猛"}),
                "max_gpu_mem / 显存上限": ("FLOAT", {"default": 14.0, "min": 8.0, "max": 16.0, "step": 0.1}),
                "fps / 帧率": ("INT", {"default": 24, "min": 1, "max": 120}),
                "qp / 输入质量": ("INT", {"default": 14, "min": 0, "max": 40,
                                         "tooltip": "输入编码质量，越小越无损 / lower = more lossless"}),
                "face_restore / 人脸修复": (["Off", "GFPGANv1.4", "CodeFormer"], {"default": "Off"}),
                "face_det_conf / 检测置信度": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95, "step": 0.05}),
                "face_blend / 融合强度": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.0, "step": 0.05}),
                "face_fidelity / 保真度": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detail_amount / 细节强度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5, "step": 0.05}),
                "detail_radius / 细节半径": ("FLOAT", {"default": 1.8, "min": 0.3, "max": 8.0, "step": 0.1}),
                "softness / 柔和度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detail_mode / 细节模式": (["classic", "smart"], {"default": "smart"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE / 图像", "width / 宽", "height / 高", "scale_used / 实际倍率", "info / 信息")
    FUNCTION = "run"
    CATEGORY = "BSAI/H3"
    DESCRIPTION = (
        "BSAI Topaz Engine Face Restore：调用本机 Topaz 神经引擎（星光 2.6 / Astra，\n"
        "默认 ComfyUI/models/Topaz_Engine）生成式放大，再叠加本插件的人脸修复\n"
        "（保真模式）+ 细节增强。底座效果对标 Topaz 官方，在其基础上继续优化\n"
        "脸部细节。Topaz engine tier: generative upscale (Starlight/Astra) +\n"
        "our fidelity-first face restore + optional detail/softness on top.\n"
        "参数名中英双语 / Parameters bilingual (EN / 中文)."
    )

    def run(self, **kw):
        g = kw.get
        images = g("images / 图像")
        model = g("model / 模型", "星光 2.6")
        scale = g("scale / 放大倍数", 2.0)
        enhancement_strength = g("enhancement_strength / 增强强度", 1.0)
        max_gpu_mem = g("max_gpu_mem / 显存上限", 14.0)
        fps = g("fps / 帧率", 24)
        qp = g("qp / 输入质量", 14)
        face_restore = g("face_restore / 人脸修复", "Off")
        face_det_conf = g("face_det_conf / 检测置信度", 0.25)
        face_blend = g("face_blend / 融合强度", 0.65)
        face_fidelity = g("face_fidelity / 保真度", 0.75)
        detail_amount = g("detail_amount / 细节强度", 0.0)
        detail_radius = g("detail_radius / 细节半径", 1.5)
        softness = g("softness / 柔和度", 0.0)
        detail_mode = g("detail_mode / 细节模式", "classic")
        t0 = time.time()
        b, h, w, c = images.shape
        if c != 3:
            raise ValueError(f"Topaz 档需要 RGB 3 通道, got {c}")
        model_id = self.TOPAZ_MODEL_IDS.get(model, "slp-26")
        frames = (images * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        out = _topaz_upscale(frames, fps, scale, enhancement_strength, max_gpu_mem, qp, model_id)
        out_t = torch.from_numpy(out.astype(np.float32) / 255.0)
        nf = 0
        if face_restore != "Off":
            out_t, nf = _face_restore_frames(out_t, face_restore, face_det_conf, face_blend, face_fidelity)
        if (detail_amount > 0 or softness > 0) and torch.cuda.is_available():
            dev = torch.cuda.current_device()
            x = out_t.to(dev).permute(0, 3, 1, 2)
            if detail_amount > 0:
                x = _detail_enhance_gpu(x, detail_amount, detail_radius, detail_mode)
            if softness > 0:
                x = _soften_gpu(x, softness)
            out_t = x.permute(0, 2, 3, 1).cpu()
        bh, bw = out_t.shape[1], out_t.shape[2]
        used = float(out_t.shape[2]) / float(w)
        elapsed = time.time() - t0
        info = (
            f"Topaz Engine [{model_id}]: {scale}x (strength={enhancement_strength}) | "
            f"output: {bw}x{bh} | scale_used: {used:.3f}x | "
            f"face: {face_restore} (blend={face_blend}, fid={face_fidelity}, faces={nf}) | "
            f"detail: {detail_amount}@{detail_radius} ({detail_mode}) | softness: {softness} | "
            f"frames: {b} | time: {elapsed:.1f}s | device: cuda"
        )
        return (out_t, bw, bh, used, info)


NODE_CLASS_MAPPINGS = {
    "BSAI_H3_Upscale4K": BSAI_H3_Upscale4K,
    "BSAI_H3_Upscale4K_Latent": BSAI_H3_Upscale4K_Latent,
    "BSAI_H3_FaceRestore": BSAI_H3_FaceRestore,
    "BSAI Topaz Engine Face Restore": BSAI_TopazEngine_FaceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_H3_Upscale4K": "BSAI H3 upscale 4K / 视频超分",
    "BSAI_H3_Upscale4K_Latent": "BSAI H3 upscale 4K Latent / H3潜空间放大",
    "BSAI_H3_FaceRestore": "BSAI H3 Face Restore / 人脸修复",
    "BSAI Topaz Engine Face Restore": "BSAI Topaz Engine Face Restore",
}

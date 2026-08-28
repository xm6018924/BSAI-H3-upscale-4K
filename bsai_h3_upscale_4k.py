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


class _CompiledWrapper(nn.Module):
    """Expose model.scale/parameters while running the torch.compile graph."""

    def __init__(self, model):
        super().__init__()
        self._m = model
        self.scale = model.scale
        self.num_block = getattr(model, "num_block", None)
        self._compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)

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
        # NOTE: no dummy warmup here. torch.compile captures the CUDA graph on the
        # FIRST real call with the actual video-frame shape; a differently-shaped
        # dummy would force a re-compile later (slower). First real run pays the
        # one-time compile cost, then every subsequent same-shape run is fast.
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


def _detail_enhance_gpu(frames, amount, radius):
    """
    frames [n,H,W,3] or [n,3,H,W] cuda -> separable-Gaussian unsharp mask, same
    shape/dtype. Shape-agnostic (accepts either layout defensively). Clamps the
    detail layer to avoid halos / overshoot on 4K video.
    """
    if amount <= 0:
        return frames
    is_chw = (frames.ndim == 4 and frames.shape[1] == 3 and frames.shape[3] != 3)
    if is_chw:
        x = frames
        n, C, H, W = x.shape
    else:
        x = frames.permute(0, 3, 1, 2)  # [n,3,H,W]
        n, C, H, W = x.shape
    sig = max(0.5, float(radius))
    ks = int(math.ceil(sig * 4)) | 1
    half = ks // 2
    dev, dt = x.device, x.dtype
    ax = torch.arange(-half, half + 1, dtype=torch.float32, device=dev)
    g = torch.exp(-(ax * ax) / (2 * sig * sig))
    g = (g / g.sum()).to(dt)
    k1 = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)  # [C,1,ks,1]
    k2 = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1)  # [C,1,1,ks]
    # depthwise separable blur: [n,3,H,W] with groups=C blurs each channel plane
    # independently with the same Gaussian kernel (never reshape-stacks channels).
    xb = F.conv2d(x, k1, padding=(half, 0), groups=C)
    xb = F.conv2d(xb, k2, padding=(0, half), groups=C)
    detail = x - xb
    out = x + amount * torch.clamp(detail, -0.3, 0.3)
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


def _video_temporal_detail(sr_cpu, lr_np, temporal_strength, detail_amount, detail_radius, scale):
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
            chunk = _detail_enhance_gpu(chunk, detail_amount, detail_radius)
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


def _restore_faces_frame(img, boxes, sess, mode, blend, fidelity=0.5):
    """Restore every detected face in one BGR/RGB uint8 frame (numpy), blend back.

    v1.4.2 fixes: (1) aspect-preserving letterbox resize into 512x512 so the
    face is never anisotropically stretched (stretched crops corrupt GFPGAN /
    CodeFormer reconstruction and cause ghosting / missing features after the
    blend); (2) elliptical blend mask centred on the face bbox instead of the
    crop centre; (3) wider pad (esp. vertically) to keep chin/forehead inside.
    """
    H, W = img.shape[:2]
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b]
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 8:
            continue
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
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((S, S, 3), dtype=np.uint8)
        dx, dy = (S - nw) // 2, (S - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        inp = canvas.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]
        feed = {sess.get_inputs()[0].name: inp}
        if len(sess.get_inputs()) > 1:  # CodeFormer: weight (fidelity)
            feed[sess.get_inputs()[1].name] = np.array([float(fidelity)], dtype=np.float64)
        o = sess.run(None, feed)[0][0].transpose(1, 2, 0)
        o = np.clip(o, 0, 1)
        o = (o * 255.0).astype(np.uint8)
        restored = o[dy:dy + nh, dx:dx + nw]            # drop letterbox padding
        restored = cv2.resize(restored, (cw, ch), interpolation=cv2.INTER_CUBIC)
        # --- elliptical mask centred on the FACE bbox (not crop centre) ---
        fx = (x1 + x2) / 2.0 - cx1
        fy = (y1 + y2) / 2.0 - cy1
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        sx = (xx - fx) / max(w / 2.0 + 0.30 * cw, 1.0)
        sy = (yy - fy) / max(h / 2.0 + 0.35 * ch, 1.0)
        m = np.clip(1.0 - np.sqrt(sx * sx + sy * sy), 0.0, 1.0)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(2.0, min(cw, ch) / 28.0))
        m = m[..., None].astype(np.float32)
        fused = blend * restored.astype(np.float32) + (1.0 - blend) * crop.astype(np.float32)
        out[cy1:cy2, cx1:cx2] = (m * fused + (1.0 - m) * crop.astype(np.float32)).astype(np.uint8)
    return out


def _face_restore_frames(out_tensor, mode, det_conf, blend, fidelity=0.5):
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
                # general-x4v3 default: ~8x faster than x4plus with nearly identical
                # quality (PSNR ~39 dB on test frames). x4plus = max quality, slowest.
                "model_name": (models, {"default": "realesr-general-x4v3.pth"}),
                # Scale supports any value 1.0-8.0 (incl. Topaz-style precise ratios
                # like 2.67 / 3.0): integer multiples of the model run directly,
                # other ratios are super-resolved to the next model integer scale and
                # then resized to the exact target (even-pixel aligned). 4 = 4K class.
                "scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.01}),
                # tile_size=0 => full-image fast path (recommended on RTX 30xx+).
                # Use a positive tile only if you run out of VRAM.
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16}),
                "tile_pad": ("INT", {"default": 16, "min": 0, "max": 128, "step": 4}),
                "batch_frames": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                # torch.compile: ~1.6-1.9x faster on fixed-size video frames.
                # One-time compile cost on first run, then cached process-wide.
                "use_compile": ("BOOLEAN", {"default": True}),
                # Temporal consistency: motion-compensated blend with neighbours
                # (Farneback optical flow on LR, GPU warp on SR). 0 = off.
                "temporal_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.8, "step": 0.05}),
                # Detail enhancement: separable-Gaussian unsharp mask on SR. 0 = off.
                "detail_amount": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.5, "step": 0.05}),
                "detail_radius": ("FLOAT", {"default": 1.5, "min": 0.3, "max": 8.0, "step": 0.1}),
                # Softness (borrowed from Topaz Starlight): after detail USM, blend
                # back a small fraction of a Gaussian-blurred copy to tame overshoot
                # and blocky artifacts. 0 = off, ~0.3 gentle, 1.0 very soft.
                "softness": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
                # Face restoration: detects faces (YOLOv8-Face) then regenerates
                # facial structure with GFPGAN / CodeFormer (ONNX, GPU, zero new
                # pip deps). Fixes H3's small / distant broken faces.
                "face_restore": (["Off", "GFPGANv1.4", "CodeFormer"], {"default": "Off"}),
                # Face detector confidence threshold (lower = more detections,
                # including tiny distant faces; may add false positives).
                "face_det_conf": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95, "step": 0.05}),
                # How strongly the restored face is blended over the original
                # crop. 1.0 = full restore, ~0.8 keeps a touch of original skin.
                "face_blend": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05}),
                # CodeFormer fidelity (0 = heavy regeneration for badly-broken
                # faces, 1 = preserve original structure/details). Ignored by
                # GFPGANv1.4.
                "face_fidelity": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE", "width", "height", "scale_used", "info")
    FUNCTION = "upscale"
    CATEGORY = "BSAI/H3"
    DESCRIPTION = (
        "H3 视频专用 AI 超分（像素域）：Real-ESRGAN 极速放大 + 光流时序一致性 + 细节增强\n"
        "+ 人脸修复（YOLOv8-Face 检测 + GFPGAN/CodeFormer，解决 H3 远景小脸崩坏模糊）。\n"
        "Tile 分块 + FP16 半精度 + torch.compile + 帧批量并行 + 模型常驻缓存。\n"
        "Video-only AI super-resolution for MiniMax H3: Real-ESRGAN extreme-speed upscale,\n"
        "with optical-flow temporal consistency, unsharp detail enhancement and\n"
        "optional face restoration (YOLOv8-Face + GFPGAN/CodeFormer) for small/distant faces."
    )

    def upscale(self, images, model_name, scale, tile_size, tile_pad, batch_frames,
                use_fp16, use_compile=True, temporal_strength=0.20,
                detail_amount=0.30, detail_radius=1.5, softness=0.30,
                face_restore="Off", face_det_conf=0.25, face_blend=0.85,
                face_fidelity=0.50):
        t0 = time.time()
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
                                     detail_radius, eff_scale)
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
            f"model: {model_name} (scale={model.scale}) | "
            f"output: {bw}x{bh} | eff_scale: {eff_scale:.3f}x | "
            f"fp16: {use_fp16} | compile: {use_compile} | tile: {tile_size} pad:{tile_pad} | "
            f"temporal: {temporal_strength} | detail: {detail_amount}@{detail_radius} | "
            f"softness: {softness} | "
            f"face: {face_restore} (conf={face_det_conf}, blend={face_blend}, fid={face_fidelity}) | "
            f"frames: {images.shape[0]} | time: {elapsed:.2f}s "
            f"(temporal+detail: {td_elapsed:.2f}s, face: {fr_elapsed:.2f}s) | "
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
                "images": ("IMAGE",),
                "face_restore": (["Off", "GFPGANv1.4", "CodeFormer"], {"default": "GFPGANv1.4"}),
                "face_det_conf": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.95, "step": 0.05}),
                "face_blend": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05}),
                "face_fidelity": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "faces_detected", "info")
    FUNCTION = "restore"
    CATEGORY = "BSAI/H3"
    DESCRIPTION = (
        "独立人脸修复节点：YOLOv8-Face 检测 + GFPGAN/CodeFormer 重建五官，\n"
        "解决 H3 中远景小脸崩坏 / 五官模糊丢失，可在任意工作流单独使用。\n"
        "Standalone face restoration for any frames: YOLOv8-Face detect +\n"
        "GFPGAN/CodeFormer regenerate, fixes small/distant broken faces."
    )

    def restore(self, images, face_restore, face_det_conf, face_blend, face_fidelity=0.50):
        t0 = time.time()
        out, nf = _face_restore_frames(images, face_restore, face_det_conf, face_blend, face_fidelity)
        info = (
            f"face restore: {face_restore} (conf={face_det_conf}, blend={face_blend}, fid={face_fidelity}) | "
            f"faces detected: {nf} | frames: {images.shape[0]} | "
            f"time: {time.time() - t0:.2f}s | "
            f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
        )
        return (out, nf, info)


NODE_CLASS_MAPPINGS = {
    "BSAI_H3_Upscale4K": BSAI_H3_Upscale4K,
    "BSAI_H3_Upscale4K_Latent": BSAI_H3_Upscale4K_Latent,
    "BSAI_H3_FaceRestore": BSAI_H3_FaceRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_H3_Upscale4K": "BSAI H3 upscale 4K / 视频超分",
    "BSAI_H3_Upscale4K_Latent": "BSAI H3 upscale 4K Latent / H3潜空间放大",
    "BSAI_H3_FaceRestore": "BSAI H3 Face Restore / 人脸修复",
}

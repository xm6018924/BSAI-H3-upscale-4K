# 🎬 BSAI H3 upscale 4K

> **MiniMax H3 专用视频高清放大超分超清极速生成插件**
> A dedicated AI video super-resolution / upscaling plugin for **MiniMax H3** — extreme-speed HD/4K upscale.

参考当前全球最先进的超分技术实现 / Built on the state-of-the-art in video super-resolution:
**Real-ESRGAN**（写实 / 动漫 / 通用）+ **Tile 分块推理** + **FP16 半精度** + **帧批量并行** + **模型常驻缓存** + **H3 专属 latent 二次采样放大**。

---

## ✨ 特性 / Features

- 🚀 **极速生成 / Extreme speed** — FP16 半精度 + Tile 分块 + 帧批量并行 + 模型进程级缓存（重复调用 **0.02s** 秒出）
- 🧠 **Real-ESRGAN 三模型** — `x4plus`（写实真人）/ `x4plus_anime_6B`（动漫）/ `general-x4v3`（通用），**自动下载**
- 🔲 **Tile + Pad 无缝分块** — 大分辨率 / 长视频不爆显存，重叠区自动加权混合消除接缝
- 🧬 **双架构自适应** — 同时支持标准 RRDBNet 与 compact（`body.N.rdb1/2/3`）结构
- 🎞️ **H3 latent 二次采样放大** — 32 像素对齐，供 H3 第二遍采样补细节（保持人物一致性）
- 🌐 **中英双语节点与文档 / Bilingual nodes & docs**
- 📦 **零额外依赖** — 仅用 PyTorch + ComfyUI 自带环境，经典 API，最大兼容性

---

## 📥 安装 / Installation

1. 把 `BSAI-H3-upscale-4K` 文件夹放进 `ComfyUI/custom_nodes/`
2. 重启 ComfyUI（首次使用选模型时**自动下载**权重到 `models/upscale_models/`，约 14–67 MB）
3. 刷新浏览器（Ctrl+F5）

> 已安装模型：放入 `ComfyUI/models/upscale_models/` 即可，插件自动识别。

---

## 🔧 节点 / Nodes

### 1️⃣ BSAI H3 upscale 4K（视频帧 → AI 超分高清帧）

| 参数 / Parameter | 说明 / Description | 默认 |
|---|---|---|
| `images` | 视频帧序列 `[B,H,W,3]`（接 H3 解码输出） | — |
| `model_name` | 超分模型（自动下载） | `realesr-general-x4v3.pth`（极速） |
| `scale` | 放大倍数 2–4 | 4 |
| `tile_size` | 分块大小（**0=不切块全图，最快**；显存不足时再调大） | 0 |
| `tile_pad` | 块重叠宽度（消除接缝） | 16 |
| `batch_frames` | 每批搬移帧数（控制显存/H2D 节奏） | 4 |
| `use_fp16` | 半精度加速 | True |
| `use_compile` | torch.compile 加速（首帧编译一次，进程内缓存） | True |

> **模型选择 / Model pick（RTX 5090 实测，960×544 → 4K，compile + FP16）**
> | 模型 | 质量 | 速度（12s/24fps 视频 ≈288 帧） |
> |---|---|---|
> | `realesr-general-x4v3.pth` ✅默认 | ≈x4plus（PSNR ~39 dB） | **最快**，约 20–36 s |
> | `RealESRGAN_x4plus_anime_6B.pth` | 良好（偏动漫锐化） | 约 77 s |
> | `RealESRGAN_x4plus.pth` | 最高 | 约 170 s |

**输出 / Outputs**: `IMAGE`（放大后帧）、`width`、`height`、`scale_used`、`info`（耗时等诊断信息）

### 2️⃣ BSAI H3 upscale 4K Latent（H3 latent → 放大 latent）

H3 专属 latent 空间放大，自动对齐 32 像素网格，用于**第二遍采样补细节**：
低分辨率生成 → 本节点放大 latent → H3 二次采样（保持原始 prompt / 参考条件 / seed）。

| 参数 | 说明 | 默认 |
|---|---|---|
| `samples` | H3 视频 latent | — |
| `upscale_method` | nearest-exact / bilinear / area / bicubic / bislerp | bilinear |
| `scale_by` | 放大倍率 | 1.5 |

**输出**: `LATENT`、`width`、`height`、`effective_scale`、`info`

---

## 🎬 H3 工作流推荐用法 / Recommended H3 workflow

**路线 A — 像素级 4K 放大（最直接）**
```
H3 生成 → VAE Decode(视频) → [BSAI H3 upscale 4K] → Save Video (4K)
```

**路线 B — latent 二次采样（H3 官方高清路线）**
```
H3 第一次采样(低分辨率 latent)
   → [BSAI H3 upscale 4K Latent](scale_by≈1.5)
   → H3 第二次采样(相同 prompt/参考条件/seed, steps 3, simple)
   → VAE Decode → Save Video
```

**路线 C — 双路组合（极致高清）**
```
H3 生成 → 二采 latent 放大(路线B) → VAE Decode → [BSAI H3 upscale 4K] → 4K
```

---

## 🧪 性能参考 / Performance (RTX 5090 Laptop, FP16 + compile)

实测（960×544 单帧 → 4K，进程内缓存命中后）：

| 模型 | ms/帧 | 288 帧（12s@24fps） |
|---|---|---|
| `realesr-general-x4v3` + compile ✅ | 76–124 | **22–36 s** |
| `RealESRGAN_x4plus_anime_6B` + compile | 267 | ~77 s |
| `RealESRGAN_x4plus` + compile | 599 | ~172 s |

> ⚠️ 关于 RTX Video Super Resolution：它是 NVIDIA 硬件级 TensorRT 实时管线（2x、流式、去噪为主），
> 纯 PyTorch 推理无法达到其 ~1s 的延迟。本插件目标为「**比 RTX VSR 更清晰的 4K 逐帧导出**」，
> 已通过 cuDNN autotune + torch.compile + 真·GPU 流水把速度从最初的 ~10.9s/8帧 提到 ~0.6–1s/8帧（约 10 倍）。

---

## 📝 更新日志 / Changelog

### v1.2.0 — 极速引擎 / Extreme-speed engine
- **修复色偏根因**：`conv_first` 后误加 LeakyReLU 导致的全局偏暗偏紫红，已移除（与 spandrel 参考逐像素一致）
- **新增 torch.compile 加速**（`use_compile` 参数，首帧编译一次、进程内缓存）
- **cuDNN benchmark 自动调优** + TF32 加速卷积
- **移除每块 `torch.cuda.empty_cache()`**（曾被每批强制清显存严重拖慢）
- **GPU 流水线优化**：结果在显存累积、按 batch 一次回传 CPU，去掉逐帧 `.cpu()` 同步卡顿
- **修复 `tile_size=0` 全图路径 bug**（此前被 `max(1,0)` 错误钳到 1 导致死循环/报错）
- **新增 SRVGGNetCompact 支持**（`realesr-general-x4v3`，残差结构 + PixelShuffle，与 spandrel maxdiff=0）
- **默认模型改为 `realesr-general-x4v3`**：与 x4plus 质量几乎一致（PSNR ~39 dB）但快 ~8 倍

### v1.0.0 — 首版 / Initial release
- 首个正式版本：像素级 Real-ESRGAN 极速超分 + H3 latent 二次采样放大
- First release: extreme-speed pixel-domain Real-ESRGAN upscale + H3 latent second-pass refine.

---

## 🧾 致谢 / Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — 超分模型架构与权重
- ComfyUI 生态 — `folder_paths` / latent 对齐约定

## ⚖️ 许可 / License

MIT — 模型权重版权归其各自作者所有 / Model weights © their respective authors.



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
- 🧑 **人脸修复 / Face restore** — YOLOv8-Face 检测 + GFPGAN / CodeFormer（ONNX·GPU·零新依赖），一键解决 H3 中远景**小脸崩坏、五官模糊丢失**
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
| `temporal_strength` | **光流时序一致性**：Farneback 光流运动补偿，与前后帧混合消除闪烁/抖动（0=关） | 0.20 |
| `detail_amount` | **细节增强**：可分高斯 USM 锐化强度（0=关） | 0.30 |
| `detail_radius` | 细节增强高斯半径（σ，越大越"粗"） | 1.5 |
| `face_restore` | **人脸修复**：YOLOv8-Face 检测 + GFPGAN/CodeFormer 重建五官，解决 H3 **远景小脸崩坏/模糊**（Off / GFPGANv1.4 / CodeFormer） | Off |
| `face_det_conf` | 人脸检测置信度（调低可检出更小的远脸，可能增加误检） | 0.25 |
| `face_blend` | 修复结果与原始融合强度（1=全替换，~0.8 保留一点原皮肤） | 0.85 |
| `face_fidelity` | **CodeFormer fidelity**（0=强重建适合严重崩坏脸，1=保留原结构细节；GFPGANv1.4 忽略） | 0.50 |

> **模型选择 / Model pick（RTX 5090 实测，960×544 → 4K，compile + FP16）**
> | 模型 | 质量 | 速度（12s/24fps 视频 ≈288 帧） |
> |---|---|---|
> | `realesr-general-x4v3.pth` ✅默认 | ≈x4plus（PSNR ~39 dB） | **最快**，约 20–36 s |
> | `RealESRGAN_x4plus_anime_6B.pth` | 良好（偏动漫锐化） | 约 77 s |
> | `RealESRGAN_x4plus.pth` | 最高 | 约 170 s |

> **时序 + 细节后处理实测（4K 输出，稳态）**
> | 配置 | 速度 |
> |---|---|
> | temporal(0.20) + detail(0.30@1.5) | ~81 ms/帧 ≈ **23 s / 288 帧** |
> | 仅 temporal | ~84 ms/帧 |
> | 仅 detail | ~49 ms/帧 |
> 效果：静态区帧间闪烁 **降 12–33%**；高频细节能量 **+60%+**（均与原始单帧超分对比）。

---

## 🔬 与主流超分技术对比 / vs. OmniSR · FlashVSR · SeedVR2 · Topaz Starlight

| 方法 | 类型 | 时序 | 实测速度(960×544→4K) | 质量 | 优点 | 缺点 |
|---|---|---|---|---|---|---|
| **本插件** | CNN/GAN | ✅(光流) | **~81ms**+超分 | 好 | 最快、时序一致、轻量、开源 | 细节重建上限低于扩散模型 |
| OmniSR | Transformer 轻量 | ❌ | 239ms | 好(PSNR高) | 0.8M 参数量 | 单帧、比 general 慢 |
| FlashVSR | 扩散一步流式 | ✅ | A100 17FPS | 高 | 时序+实时(云端) | 模型大、消费卡慢 |
| SeedVR2 | 扩散一步 | ✅ | 数秒/帧 | **最高** | 保真+时序，文本/人脸最强 | 慢、显存大 |
| Topaz Starlight | 扩散(6B) | ✅ | 慢 | 最高(专调AI视频) | AI 视频去塑料感 | 商业闭源、$799/年 |

**取舍结论**：纯 PyTorch 无法达到 NVIDIA RTX VSR 的硬件实时延迟；本插件以「**最快速度 + 光流时序一致性 + 细节增强**」在消费级 GPU 上实现逐帧 4K 导出，清晰度优于 RTX VSR 的 2x 实时增强，速度远快于 SeedVR2 / Starlight 等扩散方案。

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

### 3️⃣ BSAI H3 Face Restore（独立人脸修复，任意工作流可用）

不依赖超分链，**任何视频/图片帧**（`IMAGE`）直接修人脸：
```
H3 解码 / 任意视频帧 → [BSAI H3 Face Restore] → 修复后帧
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `images` | 任意视频/图片帧 `[B,H,W,3]` | — |
| `face_restore` | GFPGANv1.4 / CodeFormer / Off | GFPGANv1.4 |
| `face_det_conf` | 人脸检测置信度（调低检出更小远脸） | 0.25 |
| `face_blend` | 修复融合强度（1=全替换，~0.8 保留原皮肤） | 0.85 |

**输出**: `IMAGE`（修复后帧）、`faces_detected`（检测人脸总数）、`info`

> 与超分节点的 `face_restore` 参数共用同一套检测/修复引擎与模型缓存；
> 模型缺失或未启用时自动原样透传，不会崩工作流。

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

### v1.4.2 — 人脸修复质量修复 / Face-restore quality fix
- **修复「修复后五官不全 / 双眼变形 / 叠影」**（真实 H3 帧复现并解决）：
  - **根因**：旧版把非正方形的脸 crop **非等比拉伸**到 512×512 再喂 GFPGAN/CodeFormer，
    输入变形导致生成式模型重建时五官错位 → 融合后出现重影/五官缺失
  - **修复**：改为 **aspect-preserving letterbox**（保持纵横比、补边到 512²，推理后去边还原），
    重建的五官与原始位置对齐，融合无重影
  - 融合 mask 从「crop 中心径向」改为「**以人脸检测框为中心**的椭圆 mask」，
    强度峰值对准五官、向发际/背景羽化，杜绝边缘叠影
  - crop pad 加宽（横向 0.45×宽 / 纵向 0.50×高），确保下巴/额头完整入框
- **新增 `face_fidelity` 参数**（0–1，默认 0.5，仅 CodeFormer 生效）：
  0 = 强重建（适合严重崩坏脸），1 = 保留原结构细节；GFPGANv1.4 忽略该参数
- 实测（RTX 5090，真实 H3 帧 00004 崩坏脸）：修复后五官完整重建，无叠影无变形，
  单脸推理 GFPGAN ~0.33s / CodeFormer ~0.06–0.2s

### v1.4.1 — 独立人脸修复节点 / Standalone face-restore node
- **新增独立节点 `BSAI H3 Face Restore`**：任何视频/图片帧可直接单独调用
  （`IMAGE` → 修复后 `IMAGE` + 检测脸数 + 诊断信息），不依赖超分链
- 与超分节点共用同一检测/修复引擎与模型缓存；模型缺失自动透传不崩工作流

### v1.4.0 — 人脸修复 / Face restoration
- **新增人脸修复**（`face_restore` / `face_det_conf` / `face_blend`）：
  - 检测：**YOLOv8-Face**（`face_yolov8m.pt`，ultralytics，GPU）
  - 修复：**GFPGAN v1.4** / **CodeFormer**（ONNX + onnxruntime CUDA，零新增 pip 依赖——
    自动复用 torch 自带的 cudnn/cublas DLL 启用 GPU 推理）
  - 流程：超分→时序→细节后，对人脸 bbox 扩边裁剪 → 512² 重建五官 → 径向高斯 mask 无缝融合回 4K 帧
  - 实测：16×22px 极小远脸即可**重建出可辨认的眼睛/鼻子/嘴唇**；单脸 GPU 修复 ~90ms
- **解决 H3 中远景小脸崩坏 / 五官模糊丢失**：GFPGAN/CodeFormer 为生成式修复，能重建超分无法恢复的崩坏结构
- **容错**：人脸模型缺失或加载失败时自动跳过，不影响主超分

### v1.3.1 — 防御性修复 / Defensive fix
- `_detail_enhance_gpu` 形状自适应（`[n,3,H,W]` 与 `[n,H,W,3]` 均兼容），
  彻底消除 `expected input to have 3 channels` 崩溃（旧版 reshape 堆叠通道路径已删除）

### v1.3.0 — 时序一致性 + 细节增强 / Temporal consistency + detail
- **新增光流时序一致性**（`temporal_strength`）：Farneback 光流在 LR 域计算、放大到 SR，
  运动自适应权重与前后帧 warp 融合——静态区闪烁/抖动 **降 12–33%**，快速运动区自动降权防鬼影
- **新增细节增强**（`detail_amount` / `detail_radius`）：可分高斯 Unsharp Mask（GPU 批量），
  高频细节能量 **+60%+**，clamp 防过冲光晕
- **GPU 极速流水线**：fp16 全链路 + 窗口批量邻居一次性 H2D + 坐标网格缓存，
  稳态仅 **~81 ms/帧**（temporal+detail 全开，4K 输出）
- **对比调研**：与 OmniSR / FlashVSR / SeedVR2 / Topaz Starlight 横向分析——保留极速 CNN 路线，
  以时序一致性 + 细节增强补齐单帧超分在视频上的两大短板（详见 README 对比章节）

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



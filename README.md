# BSAI-H3-upscale-4K

> **English overview — full details in Chinese sections below. / 英文概览，详细中文说明见下文。**

**AI 4K Upscale plugin for MiniMax H3 video/images** (ComfyUI pixel-domain + latent-space dual channel, very fast).

One-click upscale of H3 720P videos / low-res images to 4K, with **Real-ESRGAN / FlashVSR / SeedVR2 / NVIDIA RTX / DLSS 5 / VOSR 2.0 / Topaz** engines, optical-flow temporal enhancement, face restore, detail enhancement and batch frame processing.

## Nodes

| Node | Function |
|---|---|
| **BSAI_H3_Upscale4K** | Main upscale node: **Auto-Routing** (5 preference presets: Quality / Speed / Portrait / Anime / UHD) with automatic face & resolution detection; pixel-domain engines + optical-flow temporal detail + face restore |
| **BSAI_H3_Upscale4K_Latent** | Latent-space upsampling (H3 Latent 3D Upscaler) during generation stage |
| **BSAI_H3_FaceRestore** | Face restore: GFPGAN / CodeFormer with YOLOv8-Face detection, small-face adaptive enhancement, temporal stabilization |
| **BSAI_H3_DLSS5** | DLSS 5 neural-render upscale (NVIDIA, fastest 4K path on RTX) |
| **BSAI Topaz Engine Face Restore** | Topaz commercial engine face restore |

## Engines

- **Auto-Routing**: one-click engine selection by content/input (Quality→SeedVR2 fp8 / Speed / Portrait / Anime / UHD presets + face/resolution auto-detect), inspired by HitPaw VikPea's model selector
- **Pixel-domain**: Real-ESRGAN family (x4plus / anime / general) + optical-flow temporal enhancement + batch frames
- **Diffusion video**: FlashVSR-v1.1 / SeedVR2 7B (fp8 numz path + **INT8 ComfyUI-native path** dual engines)
- **Hardware**: NVIDIA RTX Video Super Res / DLSS 5 neural rendering
- **Generative**: VOSR 2.0 (CVPR 2026, blur→clear, best for poster text restoration) + engine combos (VOSR+DLSS5 perfect, VOSR+RTX for video)
- **Face restore**: GFPGAN / CodeFormer (YOLOv8-Face detection)
- **Latent**: H3 Latent 3D Upscaler
- **Topaz**: commercial engine integration

## Key Parameters (main node)

`model_name` (incl. **auto routing**), `scale` (1-8), `tile_size`, `tile_pad`, `batch_frames`, `use_fp16`, `use_compile`, `temporal_strength`, `detail_amount`, `detail_radius`, `softness`, `detail_mode`, `face_restore`, `face_det_conf`, `face_blend`, `face_fidelity`, `face_temporal` (0.50 default), `input_adaptive` (auto), `auto_prefer` (quality/speed/portrait/anime/UHD), SeedVR2: `sv2_steps` (12), `sv2_cfg` (1.2), `sv2_sampler`, `sv2_scheduler`, `sv2_color` (lab), DLSS5: `dlss_style`/`dlss_intensity`/`dlss_detail`/`dlss_motion`.

## Quick Start

1. Copy repo into `ComfyUI/custom_nodes/BSAI-H3-upscale-4K/`
2. Install deps: `pip install numpy opencv-python spandrel onnxruntime-gpu ultralytics pillow`
3. Download models per the Chinese table below (or run `python install.py` for auto-check & core model download)
4. Restart ComfyUI, hard-refresh (Ctrl+F5)

## Example Workflow

`workflows/BSAI-H3-v2.7.0-最佳效果.json` — open in ComfyUI and run (includes sample image). Auto-routing shows its decisions in the `info` output: engine / content / preset.

## Model Weights

Full model list with download URLs (core + optional engines), legacy-download tips and directory quick-check — see the Chinese tables below. Mirror tip: replace `huggingface.co` with `hf-mirror.com` in China.

## License

Apache-2.0.

---

# BSAI-H3-upscale-4K

MiniMax H3 视频/图片 **AI 超分 4K 插件**（ComfyUI 像素域 + 潜空间双通道，速度极快）。

把 H3 生成的 720P 视频 / 低清图片一键放大到 4K，内置 **Real-ESRGAN / FlashVSR / SeedVR2 / NVIDIA RTX / DLSS 5 / VOSR 2.0 / Topaz** 多引擎，支持光流时域增强、人脸修复、细节增强、批量帧处理。

---

## 📦 模型权重清单与下载地址（必须用到的全部模型）

> 所有路径均为 **ComfyUI 根目录下的 `models/`**。标注 ✅ 的文件已随插件生态就绪/自动下载；❌ 需手动下载。
> 国内网络无法直连 HuggingFace 时，把 `huggingface.co` 换成 `hf-mirror.com` 即可。

### 一、核心模型（基础超分 + 人脸修复 + 潜空间，插件主功能必需）

| 模型 | 文件（放置路径） | 大小 | 下载地址 |
|---|---|---|---|
| Real-ESRGAN x4plus | `upscale_models/RealESRGAN_x4plus.pth` | 64 MB | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth |
| Real-ESRGAN 动漫 | `upscale_models/RealESRGAN_x4plus_anime_6B.pth` | 17 MB | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth |
| Real-ESRGAN 通用 | `upscale_models/realesr-general-x4v3.pth` | 4.7 MB | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth |
| YOLOv8-Face 人脸检测 | `ultralytics/bbox/face_yolov8m.pt`（s/n 可选） | 50 MB | https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt |
| GFPGAN 人脸修复 ONNX | `facerestore_models/GFPGANv1.4.onnx` | 325 MB | 生态流转 ONNX 导出件，无官方直链（见下方「无直链模型获取方式」） |
| CodeFormer 人脸修复 ONNX | `insightface/codeformer.onnx`（或 `facerestore_models/`） | ~350 MB | 生态流转 ONNX 导出件，无官方直链（见下方「无直链模型获取方式」） |
| MiniMax H3 Latent 3D 上采样 | `latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors` | 659 MB | https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler （仓库内含 fp16 / bf16 / fp32 三版本） |

### 二、外部引擎权重（选装对应引擎时必需）

| 引擎 | 文件（放置路径） | 大小 | 下载地址 |
|---|---|---|---|
| **FlashVSR-v1.1** | `FlashVSR-v1.1/` 整个文件夹：`LQ_proj_in.ckpt` + `TCDecoder.ckpt` + `diffusion_pytorch_model_streaming_dmd.safetensors` + `Wan2.1_VAE.pth` | 6.5 GB | https://huggingface.co/JunhaoZhuang/FlashVSR （下载整个 FlashVSR 文件夹） |
| FlashVSR 提示词嵌入 | 插件仓库内 `posi_prompt.pth`（随 git clone 自带，无需单独下载） | 4.2 MB | ✅ git clone 自带 |
| **SeedVR2 7B (fp8, numz)** | `SEEDVR2/seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | 7.9 GB | https://huggingface.co/numz/SeedVR2_comfyUI |
| SeedVR2 7B (INT8, ComfyUI 原生) | `diffusion_models/seedvr2_7b_int8_convrot.safetensors`（新引擎「SeedVR2 7B INT8」自动查找，优先于此文件，其次 `diffusion_models/` 其它 seedvr2、最后 `SEEDVR2/`） | 7.9 GB | Comfy-Org 转换包：https://huggingface.co/Comfy-Org/SeedVR2_comfyui_repackaged （ComfyUI 原生量化格式，插件走原生 NaDiT 链加载） |
| SeedVR2 VAE | `SEEDVR2/ema_vae_fp16.safetensors`（原生引擎自动查找：`SEEDVR2/` → `vae/seedvr2_ema_vae_fp16.safetensors` → `vae/ema_vae_fp16.safetensors`；三者哈希相同） | 478 MB | https://huggingface.co/numz/SeedVR2_comfyUI |
| SeedVR2 文本嵌入 | 插件仓库内 `pos_emb.pt` + `neg_emb.pt`（随 git clone 自带） | 1.3 MB | ✅ git clone 自带 |
| **DLSS 5 运行时** | `DLSS5/video2dlssnr.exe` | 429 KB | https://github.com/DaniilSokolyuk/video2dlssnr/releases/download/v1.2/video2dlssnr_release_light.zip （仅 exe） |
| DLSS 5 NVIDIA DLL | `DLSS5/nvngx_dlss.dll` + `nvngx_dlssnr.dll` + `nvngx.dll_dlssnr.dll` | 227 MB | NVIDIA 专有运行时，来自本地 DLSS5 整合包 / NVIDIA 官方；或用全量包：https://github.com/DaniilSokolyuk/video2dlssnr/releases/download/v1.2/video2dlssnr_release.zip （含 DLL，约 247 MB） |
| **VOSR 2.0** | `VOSR/` 整个仓库（含 `preset/ckpts/VOSR2/`、`Qwen-Image-vae-2d/`、`stable-diffusion-2-1-base/`、`torch_cache/` DINOv2） | ~7.5 GB | https://modelscope.cn/models/cswry/VOSR （国内直连优先）或 https://huggingface.co/cswry/VOSR ；仓库源码：https://github.com/cswry/VOSR |
| **Topaz 引擎**（商业） | `Topaz_Engine/`（neuroserver + models） | ~33 GB | 商业软件，自行部署：https://www.topazlabs.com （放入 `models/Topaz_Engine/`） |

### 三、无直链模型获取方式（GFPGANv1.4.onnx / codeformer.onnx）

这两个 ONNX 是人脸修复的社区流转导出件（插件严格要求 ONNX 格式），没有稳定官方直链，按优先级获取：

1. **已有环境复制**：装了 ComfyUI-Impact-Pack / ReActor 的机器上直接拷贝同名 `.onnx`；
2. **整合包**：BSAI / 常用 ComfyUI 整合包内 `models/facerestore_models/` 或 `models/insightface/`；
3. **自行导出**：官方 `GFPGANv1.4.pth` / `codeformer.pth`（GitHub release）用 onnx 导出脚本转换（输入 `[input]` / `[input, w]`）。

> 缺这两个文件不影响主超分功能，仅人脸修复模式的 GFPGAN/CodeFormer 选项不可用。

### 四、模型目录速查表（验证是否完整）

```
ComfyUI/models/
├── upscale_models/            RealESRGAN_x4plus.pth / _anime_6B.pth / realesr-general-x4v3.pth
├── ultralytics/bbox/          face_yolov8m.pt
├── facerestore_models/        GFPGANv1.4.onnx（+ GFPGANv1.3.onnx 备选）
├── insightface/               codeformer.onnx
├── latent_upscale_models/     minimax_h3_latent_upscaler_3d_fp16.safetensors
├── FlashVSR-v1.1/             LQ_proj_in.ckpt / TCDecoder.ckpt / DMD safetensors / Wan2.1_VAE.pth
├── SEEDVR2/                   seedvr2_ema_7b_fp8_...safetensors / ema_vae_fp16.safetensors
├── diffusion_models/          seedvr2_7b_int8_convrot.safetensors（INT8 原生引擎用）
├── vae/                       seedvr2_ema_vae_fp16.safetensors（与 SEEDVR2 内同文件）
├── DLSS5/                     video2dlssnr.exe / nvngx_dlss.dll / nvngx_dlssnr.dll / nvngx.dll_dlssnr.dll
├── VOSR/                      inference_vosr_onestep.py + preset/ckpts/（VOSR2 / Qwen VAE / SD2.1 VAE / torch_cache）
└── Topaz_Engine/              neuroserver171/neuroserver.exe + models/
```

---

## 🧪 示例工作流（v2.7.0 最佳效果）

`workflows/BSAI-H3-v2.7.0-最佳效果.json` —— 一键体验本次升级的三项新能力，**按最佳效果配置**：

| 节点 | 配置 | 说明 |
|---|---|---|
| LoadImage | `bsai_h3_v27_example.png`（已放入 `ComfyUI/input/`） | 400×532 低清人像示例，换成你自己的图/视频帧即可 |
| **BSAI_H3_Upscale4K** | 模型 = **自动路由**，偏好 = **质量优先** | 自动探测 SeedVR2 fp8（精度最高）→ INT8 → VOSR → FlashVSR → Real-ESRGAN，逐级回退 |
| ↑ 自动内容感知 | `face_restore = Off`（不用手动开） | 路由检测到人脸 → **自动开启 CodeFormer 修复 + 时域稳定** |
| ↑ 自动分辨率感知 | `input_adaptive = 自动` | 低清输入自动 HD 增强重建；≥720P 输入自动 UHD 保守细节 |
| ↑ 时域稳定 | `face_temporal = 0.50` | 跨帧跟踪 + 参数 EMA，防"框抖动→强度跳变"闪烁 |
| ↑ 画质 | `temporal 0.2 / detail smart 0.5 / softness 0.1 / scale 4.0` | 光流时序 + 智能细节重建 + 柔和收敛，4K 输出 |
| PreviewImage / SaveImage | — | 左下预览原图、右侧保存 4K 结果（`BSAI-H3-v27-4K` 前缀） |

使用：拖入 ComfyUI → 点运行（LoadImage 可换图）。info 输出会完整显示自动路由决策（引擎 / 内容 / 档位），可直观看到 v2.7.0 的"自动"如何工作。

---

## 🛠️ v2.7.1 修复（VOSR / DLSS5 大图健壮性）

| 问题 | 根因 | 修复 |
| --- | --- | --- |
| VOSR 大图 OOM | 非方形帧被补成正方形（1536×2048→2048×2048，面积+33%）后全图 VAE 推理 | 短边 >1024 自动启用官方 latent 分块（tile 640）+ VAE 分块（1024），峰值显存降到单 tile 级；tile 路径只做最小 16 对齐不再补正方形 |
| VOSR `Input height (131) should be divisible by patch size (2)` | VOSR DiT latent=像素/8、patch=2 → 输入必须为 16 的倍数（507×524 补成 524 → latent 131 非偶） | fast path 补成"正方形+16 对齐"，tile 路径最小 16 对齐，推理后按原比例裁回 |
| DLSS5 `CreateFeature 18 failed` | DLSS NR 输出上限 7680（实测 8192 失败） | 目标超限自动分块（每块 ≤7680）+ 重叠线性融合，任意尺寸可用 |
| VOSR 子进程偶发崩溃（WinError 10060） | torch.hub 每次启动联网探测 GitHub 默认分支，网络超时未捕获 | VOSR 仓库 `inference_vosr_onestep.py` / `inference_vosr.py` 打离线 DINOv2 补丁：本地权重 + 缓存 repo 加载，失败才回退在线 |

> **VOSR 离线补丁部署**（换机/重装时需要）：把补丁应用到 `models/VOSR/inference_vosr_onestep.py` 与 `inference_vosr.py` 的 `load_dinov2()` —— 本地优先从 `preset/ckpts/torch_cache/checkpoints/dinov2_vitl14_pretrain.pth` + `preset/ckpts/torch_cache/facebookresearch_dinov2_main/` 加载，彻底免除 GitHub 联网依赖。

实测（RTX 5090 Laptop 24GB）：507×524→1014×1048（42s）；1536×2048→4096×3072（tile，77s）；VOSR×2+DLSS5×2 全链→**8192×6144**（83.5s）；22 项单元测试全过。

---

## 🛠️ DLSS5 报错「未找到 video2dlssnr.exe」排错 / Troubleshooting (bilingual)

**现象 / Symptom**：节点 130 执行报 `RuntimeError: [BSAI-H3/DLSS5] 未找到 video2dlssnr.exe`。
DLSS 5 引擎是"本机调用"设计：exe 与 3 个 NVIDIA 专有 DLL **不随插件分发**，需就位于 `ComfyUI/models/DLSS5/`。
The DLSS 5 engine invokes a local runtime: the exe + 3 proprietary NVIDIA DLLs are **not shipped with the plugin** and must exist under `ComfyUI/models/DLSS5/`.

**所需 4 个文件 / Required files (put all 4 in `ComfyUI/models/DLSS5/`)**：

| 文件 file | 大小 size | 来源 source |
|---|---|---|
| `video2dlssnr.exe` | 0.4 MB | 官方 light 包 official light zip: [video2dlssnr_release_light.zip](https://github.com/DaniilSokolyuk/video2dlssnr/releases/download/v1.2/video2dlssnr_release_light.zip) |
| `nvngx_dlss.dll` | 56 MB | NVIDIA 专有 proprietary：驱动包 driver pack / 全量包 full zip / 本机其它整合包 other local packs |
| `nvngx_dlssnr.dll` | 158 MB | 同上 same as above |
| `nvngx.dll_dlssnr.dll` | 13 KB | 同上 same as above |

> 全量包（exe+DLL 一次到位，约 247MB）full bundle: [video2dlssnr_release.zip](https://github.com/DaniilSokolyuk/video2dlssnr/releases/download/v1.2/video2dlssnr_release.zip)

**自动发现（v2.9.3 起）/ Auto-discovery (since v2.9.3)**：运行时查找不再只扫描写死的 `C:\BSAI\DLSS5`——插件与 `install.py` 会从**实际安装路径**向外发现同层其它 ComfyUI 整合包的 `models/DLSS5/`（例如 `G:\Comfyui_BSAI\ComfyUI\models\DLSS5\`），命中即自动复制/直接使用，无需任何手动操作；旧默认目录与环境变量 `VIDEO2DLSSNR_PACK_DIR` / `VIDEO2DLSSNR_EXE` 仍然有效。
The search area is now derived from the live install path: sibling ComfyUI packs' `models/DLSS5/` on the same drive/level are probed automatically (e.g. `G:\Comfyui_BSAI\ComfyUI\models\DLSS5\`) and reused without any manual step; the legacy `C:\BSAI\DLSS5` default and the `VIDEO2DLSSNR_PACK_DIR` / `VIDEO2DLSSNR_EXE` env vars keep working.

**手动兜底 / Manual fallback**：
1. 运行 `python install.py`（自动扫描本地包复制，缺 exe 时联网下载官方 light 包 / auto-copies local packs, downloads the official light zip if exe missing）；
2. 或手动把 4 个文件放入 `ComfyUI/models/DLSS5/`（或放入本插件 `bin/` 只放 exe，DLL 仍需同目录）/ or copy the 4 files into `ComfyUI/models/DLSS5/` (exe-only into plugin `bin/` works too, DLLs must sit beside it)。

---

## 🚀 快速安装

1. 把本仓库放入 `ComfyUI/custom_nodes/BSAI-H3-upscale-4K/`；
2. 安装依赖：`pip install numpy opencv-python spandrel onnxruntime-gpu ultralytics pillow`；
3. 下载上表模型放入对应 `models/` 目录（或直接运行 `python install.py` 一键自检 + 自动下载核心模型）；
4. 重启 ComfyUI，刷新浏览器（Ctrl+F5）。

---

## 🎛️ 功能与引擎

- **自动路由**：按内容/输入一键选引擎（质量/速度/人像/动漫/UHD 五档偏好 + 人脸/分辨率自动感知），对标 HitPaw VikPea 模型选择器
- **像素域超分**：Real-ESRGAN 系（x4plus / anime / general）+ 光流时域增强 + 批量帧
- **扩散视频超分**：FlashVSR-v1.1 / SeedVR2 7B（fp8 numz 路径 + **INT8 ComfyUI 原生路径**双引擎）
- **硬件超分**：NVIDIA RTX Video Super Res / DLSS 5 神经渲染（RTX 显卡最快 4K 路线）
- **生成式超分**：VOSR 2.0（CVPR 2026，模糊图→清晰，海报文字还原最强）+ 双引擎组合
  - `VOSR 2.0 + DLSS 5（双引擎完美档）`：VOSR 生成细节 → DLSS 硬件放大
  - `VOSR 2.0 + RTX（视频推荐两级放大）`：VOSR 去模糊 → RTX 二次清晰化
- **人脸修复**：GFPGAN / CodeFormer（YOLOv8-Face 检测，小脸自适应增强）
- **潜空间上采样**：H3 Latent 3D Upscaler（生成阶段直接放大）
- **Topaz 引擎**：商业软件接入

---

## 📝 更新日志

### v2.9.3 — DLSS5 本地引擎按安装路径自动发现（修复「未找到 video2dlssnr.exe」）/ Install-path-aware DLSS5 auto-discovery
- **背景 / Context**：用户报节点执行 `RuntimeError: [BSAI-H3/DLSS5] 未找到 video2dlssnr.exe`。运行时其实已存在于本机另一整合包 `G:\Comfyui_BSAI\ComfyUI\models\DLSS5\`（4 文件齐全），但插件与 install.py 的本地包扫描只写死了 `C:\BSAI\DLSS5`，跨盘/跨目录的既有文件发现不了。
  The runtime already existed in a sibling pack on another drive, but both the node and `install.py` only scanned the hard-coded `C:\BSAI\DLSS5`.
- **修复 / Fix**：查找根改为从**插件实际安装路径**推导——自动探测同层（及历史 `C:\BSAI` 树）名称含 comfy/bsai/aki 的兄弟整合包的 `ComfyUI/models/DLSS5`、`models/DLSS5`、`DLSS5` 三种布局（精确拼路径探测，不做全盘遍历）；命中 exe 或 zip 即直接使用/自动解包复制。环境变量 `VIDEO2DLSSNR_PACK_DIR` / `VIDEO2DLSSNR_EXE` 与旧默认路径保持兼容优先。
  Search roots are now derived from the live install path: sibling packs (same level, plus the legacy `C:\BSAI` tree) are probed at the three known layouts — exact-path checks only, no full-disk walk. Env vars and the legacy default keep working.
- **文档 / Docs**：缺件报错信息升级为中英双语并完整列出 4 个所需文件（exe 0.4MB + nvngx_dlss 56MB + nvngx_dlssnr 158MB + nvngx.dll_dlssnr 13KB）及获取途径；README 新增双语排错章节；install.py 待办提示双语化。
  The missing-runtime error is now bilingual with the exact 4-file checklist and sources; README gained a bilingual troubleshooting section; install.py hints are bilingual.

### v2.8.1 — UI 布局：自动路由偏好移至模型选择之上
- `auto_prefer / 自动路由偏好` 参数从参数区底部移到节点**第一栏（模型选择之上）**：先选档位偏好再看引擎，符合"自动路由"的操作直觉；仅当 model_name = 自动路由 时生效（行为不变）

### v2.8.0 — 质量优先档画质重构 + 单帧崩溃修复 + SeedVR2 fp8 路径修复

**背景**：用户反馈 v2.7.1 按"自动路由/质量优先"处理 507×524 单帧输出画质"不堪入目"（眼/唇细节劣化）；手动切 Real-ESRGAN 模型后节点崩溃 `AttributeError: 'NoneType' object has no attribute 'shape'`。

**① 单帧崩溃修复（问题②）**：根因是 `upscale()` 仅 Real-ESRGAN 分支在多帧且 cv2 可用时才给 `lr_np` 赋值，其余引擎分支恒为 `None`；单帧输入时 `temporal_strength=0.2>0` 仍进入 `_video_temporal_detail` → `_compute_flow_pairs(None).shape` 崩溃。
- `_compute_flow_pairs`：`lr_np is None` 或 `shape[0]<2` 直接返回 `[]`
- `_video_temporal_detail`：单帧/无光流输入自动 `temporal_strength=0`（跳过 flow）
- Real-ESRGAN 分支：单帧或无 cv2 时 `temporal_strength=0`——**任何引擎 + 任何输入尺寸都不再崩溃**

**② 质量优先档画质重构（问题①）**：本地对照实验（同一张 507×524 原图）实测——
| 路径 | 输出 | 耗时 | Laplacian |
|---|---|---|---|
| SeedVR2 INT8 8步 cfg1.0（旧默认） | 1048×1014 | 130s | 375 |
| SeedVR2 fp8 官方默认（修复后） | 1048×1014 | **84s** | 290 |

结论：**根因是路由选中 INT8 量化权重 + 8 步低采样**（w4a4 量化精度损失在眼唇等细节区最明显，高频被过度强调产生伪影/失真），而非后处理参数。
- **路由换序**：质量优先 = SeedVR2 **fp8（精度最高）** → INT8 → VOSR → FlashVSR → Real-ESRGAN；fp8 已实测更快（84s vs 130s）且权重精度更高
- **INT8 采样增强**：仅当回退到 INT8 时自动覆盖 `sv2_steps=16 / sv2_cfg=1.5`（补偿量化损失），信息写入 info 输出
- **人脸自动轻修复**：自动检测到人脸时，CodeFormer 保真 0.6→**0.45**、融合 0.7→**0.55**（只补细节不改脸型，杜绝"塑料感"）
- **HD 增强档防过度锐化**：低清输入细节强度上限 1.0→**0.6**
- **节点默认值**：`sv2_steps` 8→**12**、`sv2_cfg` 1.0→**1.2**

**③ SeedVR2 fp8 路径修复（潜伏 bug）**：`_seedvr2_upscale` 从未真正输出放大图——官方插件返回 `io.NodeOutput(sample)`（`.result` 为 args 元组），旧解包逻辑误判为非 tensor 直接回退原图。修复为正确解包 `NodeOutput.result[0]`，fp8 路径首次真实可用（此前若被手动选中会"输出尺寸不变"）。

实测：24 项单元测试全过（单帧 temporal 防御 / fp8 优先路由 / INT8 采样覆盖 / 人脸轻修复 / HD 档上限 / 节点默认值 / fp8 返回解包）；节点级 E2E auto/质量优先 507×524→1048×1014 正常出图。

### v2.7.0 — HitPaw 技术路线本地落地：自动路由 + UHD 输入自适应 + 人脸时域稳定
- **背景**：应要求全球调研 HitPaw VikPea 超分技术（官方模型文档 / developer.hitpaw.com API / 官方 OpenClaw skill）。结论：VikPea 为闭源商业软件 + 云端推理，**模型权重不可提取**；唯一合法接入路径是官方云端 REST API（付费积分、输入须公网 URL、分钟~小时级异步），与本插件"本地帧级批处理 + 离线 + 免费"定位根本冲突 → **采纳其技术理念本地落地（路径 B）**，对标映射：
  - `ultrahd_restore_2x`（UHD 超分，高清输入保守细节）→ **输入自适应**
  - `portrait_restore`（多帧融合 + 时空对齐防闪烁）→ **人脸时域稳定**
  - VikPea 多模型选择器（General/Animation/Portrait/UHD/Generative）→ **自动路由**
- **① 模型自动路由**：新增引擎「自动路由 (按内容/输入推荐)」，零新依赖（只探测已有权重 + 抽样人脸检测）：
  - 5 档偏好 `auto_prefer / 自动路由偏好`：质量优先（SeedVR2 INT8→fp8→VOSR→FlashVSR→Real-ESRGAN 逐级回退）、速度优先、人像优先（强制 CodeFormer 人脸修复）、动漫优先（anime 权重）、UHD 优先
  - 自动内容感知：YOLO 抽样检测到人脸 → 自动开启 CodeFormer 人脸修复 + 时域稳定（无需手动开关）
  - 自动分辨率感知：≥720P 高清输入 → UHD 保守细节；<540P 低清输入 → HD 增强细节
  - 路由决策完整写入 info 输出（引擎/内容/档位全透明）
- **② UHD 输入自适应**：新参数 `input_adaptive / 输入自适应`（自动/关/UHD 保守档/HD 增强档）：≥720P 时细节强度打 6 折（上限 0.35）+ 强制 classic 模式（防过度锐化/伪影）+ 柔和度下限 0.15；<720P 时细节增强 1.2 倍配合 smart 重建。对标 Ultra HD 模型的"自然无数码感"定位
- **③ 人脸时域稳定**：新参数 `face_temporal / 人脸时域稳定`（默认 0.50，主节点 / 独立 FaceRestore / Topaz FaceRestore 三节点同步）：
  - 跟踪增强：IoU 断链时按中心距离 + 尺寸比 gating 续链（快速移动/缩放不再断链重检）
  - **参数级 EMA**：每条人脸轨迹的融合强度/保真度做跨帧指数平滑（0=关闭，等同 v2.6 行为；0.5=推荐；1.0=完全跟随历史）——消除"检测框抖动 → 修复强度逐帧跳变"的闪烁，同时保持逐帧像素级恢复（不混合结果像素，规避 v2.3.1 曾出现的鬼影）
- 实测：22 项单元测试全过（参数 EMA 跨断点抖动跳变 0.0105→0.0053 减半；dist-gate 续链；路由 5 偏好 + 人脸/分辨率感知）；端到端 4 帧真实人像（CodeFormer 修复正常、无伪影）；主节点完整链路 2 帧 1024×1365→2048×2730 30.3s，auto 路由正确选中引擎 + 自动开人脸修复 + UHD 保守档
- 兼容性：`face_temporal=0` / `input_adaptive=关` / 不选自动路由时行为与 v2.6.0 完全一致

### v2.6.0 — SeedVR2 7B INT8 (ComfyUI 原生) 引擎 + 模型路径自动加载
- **背景**：用户新下载 `models/diffusion_models/seedvr2_7b_int8_convrot.safetensors`（Comfy-Org INT8 量化，7.9GB）与 `models/vae/seedvr2_ema_vae_fp16.safetensors`
- **兼容性核查结论**：VAE 与现有 `SEEDVR2/ema_vae_fp16.safetensors` **哈希完全一致**（同一文件）；INT8 权重含 `comfy_quant`/`weight_scale` + I8/U8 量化层（288 层），**numz 插件加载器无法读取**（strict=False 会静默丢权重），但 **ComfyUI 原生链（comfy.sd.load_diffusion_model + quant_ops）完整支持**（实测识别为 NaDiT 8.24B 全量加载）
- **新增引擎**「SeedVR2 7B INT8 (ComfyUI原生)」：走 ComfyUI 官方 NaDiT + KSampler 采样链，推理流程 1:1 对齐 `comfy_extras/nodes_seedvr.py`（pad16 + 补帧 4n+1 → VAE tiled encode → conditioning → sample → tiled decode → lab/wavelet/adain 颜色校正）
- **模型路径自动加载**：DIT 自动查找 `diffusion_models/seedvr2_7b_int8_convrot.safetensors`（INT8 优先）→ `diffusion_models/` 其它 seedvr2 → `SEEDVR2/` fp8；VAE 自动查找 `SEEDVR2/` → `vae/seedvr2_ema_vae_fp16.safetensors` → `vae/ema_vae_fp16.safetensors`
- 新参数：`sv2_steps / SeedVR2步数`（默认 8）、`sv2_cfg / SeedVR2保真度`（默认 1.0）、`sv2_sampler / SeedVR2采样器`、`sv2_scheduler / SeedVR2调度器`、`sv2_color / SeedVR2色彩校正`（默认 lab）
- 工程细节：采样后主动卸载 7B DiT 释放显存，VAE encode/decode 显式 256 tile（避免整张 OOM 回退 32x32 超慢 tile），tiled 阶段包 `torch.no_grad()`（规避 seedvr tiled_vae 的 inplace 视图冲突）
- 实测：320×448 2 帧 → 640×896（8 步）32.7s，PSNR 31.0dB vs 双三次（结构完整保留）

### v2.5.0 — 结合视频实测优化：VOSR 通道修复 + DLSS5 参数对齐 + VOSR+RTX 组合
- **依据**：B 站《低配必看！VOSR 遇上 DLSS5：如何用超分放大技术白嫖极致画质？》（BV1WBYV6hEXa）实测结论落地
- DLSS5 默认参数对齐视频实测：`color / 色彩` 1.0→**0.0**（视频实测开 1 皮肤油腻）、`skin / 皮肤` -1.0→**0.0**（保留肌理）、`detail` 保持 1.0（建议 1 或更高）
- 修复 VOSR 通道必然失败 bug：`inference_vosr_onestep.py` 不支持 `--cfg_scale`，现动态检测；修复 `shutil` 缺失 NameError；**修复非方形输入静默失败**（自动 reflect 补边成方形 → 推理 → 裁回原比例）
- 新增 `VOSR 2.0 + RTX (视频推荐两级放大)` 引擎（新参数 `rtx_chain_scale / RTX串联倍率`）
- VOSR 仓库支持 `ComfyUI/models/VOSR` 路径（`VOSR_HOME` → `models/VOSR` → `C:\BSAI\VOSR` 依次查找）

### v2.4.0 — DLSS 5 神经渲染超分引擎 + 2026 最新超分技术全景
- 新增 DLSS 5 (NVIDIA 神经渲染) 引擎、VOSR 2.0 生成式超分、RTX 硬件超分、FlashVSR/SeedVR2 扩散视频超分、Topaz 商业引擎接入

### 更早版本
- 光流时域增强、人脸修复（GFPGAN/CodeFormer + YOLOv8-Face）、批量帧处理、Latent 3D 上采样等

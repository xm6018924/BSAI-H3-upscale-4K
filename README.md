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
| **SeedVR2 7B** | `SEEDVR2/seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | 7.9 GB | https://huggingface.co/numz/SeedVR2_comfyUI |
| SeedVR2 VAE | `SEEDVR2/ema_vae_fp16.safetensors` | 478 MB | https://huggingface.co/numz/SeedVR2_comfyUI |
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
├── DLSS5/                     video2dlssnr.exe / nvngx_dlss.dll / nvngx_dlssnr.dll / nvngx.dll_dlssnr.dll
├── VOSR/                      inference_vosr_onestep.py + preset/ckpts/（VOSR2 / Qwen VAE / SD2.1 VAE / torch_cache）
└── Topaz_Engine/              neuroserver171/neuroserver.exe + models/
```

---

## 🚀 快速安装

1. 把本仓库放入 `ComfyUI/custom_nodes/BSAI-H3-upscale-4K/`；
2. 安装依赖：`pip install numpy opencv-python spandrel onnxruntime-gpu ultralytics pillow`；
3. 下载上表模型放入对应 `models/` 目录（或直接运行 `python install.py` 一键自检 + 自动下载核心模型）；
4. 重启 ComfyUI，刷新浏览器（Ctrl+F5）。

---

## 🎛️ 功能与引擎

- **像素域超分**：Real-ESRGAN 系（x4plus / anime / general）+ 光流时域增强 + 批量帧
- **扩散视频超分**：FlashVSR-v1.1 / SeedVR2 7B
- **硬件超分**：NVIDIA RTX Video Super Res / DLSS 5 神经渲染（RTX 显卡最快 4K 路线）
- **生成式超分**：VOSR 2.0（CVPR 2026，模糊图→清晰，海报文字还原最强）+ 双引擎组合
  - `VOSR 2.0 + DLSS 5（双引擎完美档）`：VOSR 生成细节 → DLSS 硬件放大
  - `VOSR 2.0 + RTX（视频推荐两级放大）`：VOSR 去模糊 → RTX 二次清晰化
- **人脸修复**：GFPGAN / CodeFormer（YOLOv8-Face 检测，小脸自适应增强）
- **潜空间上采样**：H3 Latent 3D Upscaler（生成阶段直接放大）
- **Topaz 引擎**：商业软件接入

---

## 📝 更新日志

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

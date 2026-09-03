#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSAI-H3-upscale-4K — 一键环境自检 + 自动补齐安装器
====================================================
在"空电脑"上部署本插件时运行：

    python install.py

已有依赖 -> 自动跳过；缺失 -> 静默自动安装 / 下载。
结束后输出一张 ✅/⚠️/❌ 汇总表，❌ 项会给出具体待办与放置路径。

检测与补齐范围
  1. Python / ComfyUI 根目录定位
  2. PyTorch + CUDA + GPU 型号（DLSS5 需要 NVIDIA RTX；非 NVIDIA 也可用 Real-ESRGAN）
  3. Python 依赖包：numpy / opencv-python / spandrel / onnxruntime-gpu / ultralytics / pillow
  4. 超分模型：Real-ESRGAN 三模型（官方 GitHub release，自动下载）
  5. 人脸修复模型：GFPGANv1.4 / CodeFormer / YOLOv8-Face
  6. 外部引擎插件：FlashVSR / SeedVR2 / NVIDIA RTX（git clone 到 custom_nodes，可选）
  7. DLSS5 运行时：本地已有整合包 -> 自动复制；缺失 -> 给出获取指引
  8. Topaz 引擎：商业软件，仅检测提示，不自动获取

可选参数
  --no-pip        跳过 Python 依赖包安装
  --no-models     跳过模型下载
  --no-git        跳过外部引擎插件 clone
  --dlss-dir DIR  额外指定含 video2dlssnr.exe 的目录（可重复指定）
  --dlss5-full    自动下载 video2dlssnr 官方全量 release（含 NVIDIA DLL，约 247MB）
"""
import io
import os
import re
import shutil
import subprocess
import sys
import urllib.request

TAG = "[BSAI-INSTALL]"
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
HERE = os.getcwd()

# ---------------------------------------------------------------------------
# 基础定位
# ---------------------------------------------------------------------------
def comfyui_root():
    """脚本位于 <root>/custom_nodes/BSAI-H3-upscale-4K/ -> root 为其上两级。"""
    p = os.path.dirname(os.path.dirname(PLUGIN_DIR))
    if os.path.basename(p) == "custom_nodes":
        return os.path.dirname(p)
    return p


def models_dir():
    return os.path.join(comfyui_root(), "models")


def custom_nodes_dir():
    return os.path.join(comfyui_root(), "custom_nodes")


def find_pythons():
    """返回候选 Python 解释器（优先 ComfyUI 自带的）。"""
    cands = []
    root = comfyui_root()
    for rel in ("python_embeded/python.exe", "python/python.exe", "python.exe"):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            cands.append(p)
    exe = getattr(sys, "executable", None)
    if exe and exe not in cands:
        cands.append(exe)
    return cands


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def run(cmd, **kw):
    kw.setdefault("timeout", 1800)
    kw.setdefault("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0))
    kw["stdout"] = subprocess.PIPE
    kw["stderr"] = subprocess.STDOUT
    kw["encoding"] = "utf-8"
    kw["errors"] = "replace"
    return subprocess.run(cmd, **kw)


def download(url, dest, label, timeout=120):
    """静默下载到 dest（先写 .part 再原子替换）。已存在且非空则跳过。"""
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"{TAG} [跳过] 已有 {label}: {dest}")
        return True
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 BSAI-H3-installer"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        if os.path.getsize(tmp) == 0:
            os.remove(tmp)
            print(f"{TAG} [失败] {label} 下载为空: {url}")
            return False
        os.replace(tmp, dest)
        print(f"{TAG} [下载] {label} -> {dest} ({os.path.getsize(dest)//1024} KB)")
        return True
    except Exception as e:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        print(f"{TAG} [失败] {label}: {e}\n         URL: {url}")
        return False


def git_available():
    return shutil.which("git") is not None


# ---------------------------------------------------------------------------
# 1) Python / ComfyUI
# ---------------------------------------------------------------------------
def check_python():
    print(f"\n{TAG} == 1/8 Python 与 ComfyUI ==")
    ok = True
    v = sys.version_info
    print(f"{TAG} 解释器: {sys.executable} | Python {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(f"{TAG} [警告] Python >= 3.10 建议（当前 {v.major}.{v.minor}）")
    root = comfyui_root()
    if not os.path.isdir(root) or not os.path.isdir(os.path.join(root, "custom_nodes")):
        print(f"{TAG} [失败] 未定位到 ComfyUI 根目录（期望 {root}/custom_nodes 存在）")
        ok = False
    else:
        print(f"{TAG} ComfyUI 根目录: {root}")
    pys = find_pythons()
    if pys:
        print(f"{TAG} 可用 Python: {pys[0]}")
    return ok


# ---------------------------------------------------------------------------
# 2) PyTorch / CUDA / GPU
# ---------------------------------------------------------------------------
def check_torch():
    print(f"\n{TAG} == 2/8 PyTorch / CUDA / GPU ==")
    try:
        import torch
        print(f"{TAG} torch {torch.__version__} | cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            rt = torch.version.cuda
            print(f"{TAG} GPU: {name} | SM {cap[0]}.{cap[1]} | CUDA {rt}")
            nvidia = "nvidia" in name.lower()
            rtx = ("rtx" in name.lower()) or ("nvidia" in name.lower() and cap[0] >= 8)
            print(f"{TAG} NVIDIA RTX (支持 DLSS5): {'是' if rtx else '否/未知'}")
            return True
        print(f"{TAG} [警告] torch 存在但 CUDA 不可用（DLSS5 / RTX 需 CUDA GPU）")
        return False
    except Exception as e:
        print(f"{TAG} [失败] 未检测到 torch：{e}")
        print(f"{TAG}         请先安装 ComfyUI（其自带 PyTorch+cuDNN）；裸装 torch 易选错 CUDA 版本。")
        return False


# ---------------------------------------------------------------------------
# 3) Python 依赖包（静默 pip install）
# ---------------------------------------------------------------------------
# (pip 包名, 导入用模块名) —— 模块名与包名不一致的必须显式给出，否则误判缺失重复安装
PIP_PACKAGES = [
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("spandrel", "spandrel"),
    ("onnxruntime-gpu", "onnxruntime"),
    ("ultralytics", "ultralytics"),
    ("pillow", "PIL"),
]


def check_pip(no_pip):
    print(f"\n{TAG} == 3/8 Python 依赖包 ==")
    if no_pip:
        print(f"{TAG} [跳过] --no-pip")
        return True
    missing = []
    for pkg, mod in PIP_PACKAGES:
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if not missing:
        print(f"{TAG} 全部依赖已就绪: {', '.join(p for p, _ in PIP_PACKAGES)}")
        return True
    print(f"{TAG} 缺失 {len(missing)} 个依赖，静默安装: {', '.join(missing)}")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           "--disable-pip-version-check", "--no-warn-script-location"] + missing
    r = run(cmd)
    ok = r.returncode == 0
    if not ok:
        print(f"{TAG} [失败] pip 安装失败（rc={r.returncode}），可手动执行:\n"
              f"         {sys.executable} -m pip install {' '.join(missing)}")
        return False
    print(f"{TAG} 已安装: {', '.join(missing)}")
    return True


# ---------------------------------------------------------------------------
# 4) Real-ESRGAN 超分模型（自动下载）
# ---------------------------------------------------------------------------
ESRGAN_MODELS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}


def check_esrgan(no_models):
    print(f"\n{TAG} == 4/8 Real-ESRGAN 超分模型 ==")
    if no_models:
        print(f"{TAG} [跳过] --no-models")
        return True
    d = os.path.join(models_dir(), "upscale_models")
    os.makedirs(d, exist_ok=True)
    all_ok = True
    for name, url in ESRGAN_MODELS.items():
        ok = download(url, os.path.join(d, name), f"模型 {name}")
        all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# 5) 人脸修复模型（YOLOv8-Face 自动下载；GFPGAN/CodeFormer ONNX 手动）
# ---------------------------------------------------------------------------
# GFPGANv1.4.onnx / codeformer.onnx 是生态内流转的 ONNX 导出件，无稳定官方直链，
# 且插件严格要求 ONNX 格式 —— 自动下载易拿到错误格式文件，故列为手动项（来源见提示）。
FACE_YOLO_URLS = [
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt",
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
]


def check_face(no_models):
    print(f"\n{TAG} == 5/8 人脸修复模型（YOLOv8-Face 自动 / GFPGAN·CodeFormer 手动）==")
    if no_models:
        print(f"{TAG} [跳过] --no-models")
        return True
    d = os.path.join(models_dir(), "ultralytics", "bbox")
    os.makedirs(d, exist_ok=True)
    # 已有任一 face_yolov8*.pt 即跳过
    have = [fn for fn in os.listdir(d) if fn.startswith("face_yolov8") and fn.endswith(".pt")]
    if have:
        print(f"{TAG} [跳过] 已有 {have[0]}")
        return True
    ok = False
    for url in FACE_YOLO_URLS:
        name = os.path.basename(url)
        if download(url, os.path.join(d, name), f"人脸检测 {name}", timeout=120):
            # .pt 合理性检查：ultralytics 权重通常 >= 20MB
            size = os.path.getsize(os.path.join(d, name))
            if size < 20 * 1024 * 1024:
                print(f"{TAG} [警告] {name} 尺寸异常（{size//1024}KB），可能非完整权重，删除")
                os.remove(os.path.join(d, name))
                continue
            ok = True
            break
    if not ok:
        print(f"{TAG} [待办] 人脸检测模型下载失败（网络/镜像不可达），可手动放入 {d}:")
        print(f"{TAG}         face_yolov8m.pt 来源: https://huggingface.co/Bingsu/adetailer")
    # GFPGAN / CodeFormer ONNX 手动项
    fr = os.path.join(models_dir(), "facerestore_models")
    need_onnx = [n for n in ("GFPGANv1.4.onnx", "codeformer.onnx")
                 if not os.path.isfile(os.path.join(fr, n))]
    if need_onnx:
        print(f"{TAG} [待办] 人脸修复 ONNX（非核心，仅人脸修复功能需要）缺: {', '.join(need_onnx)}")
        print(f"{TAG}         请放入 {fr}: 来源 = 已装 ComfyUI-Impact-Pack 的机器 / 网盘中的")
        print(f"{TAG}         GFPGANv1.4.onnx 与 codeformer.onnx（本插件人脸修复需要 ONNX 格式）")
        return False
    return ok


# ---------------------------------------------------------------------------
# 6) 外部引擎插件（FlashVSR / SeedVR2 / RTX Nodes，git clone）
# ---------------------------------------------------------------------------
ENGINE_PLUGINS = [
    ("ComfyUI-FlashVSR_Ultra_Fast", "https://github.com/liusida/ComfyUI-FlashVSR_Ultra_Fast.git"),
    ("ComfyUI-SeedVR2_VideoUpscaler", "https://github.com/abanob-magdy/ComfyUI-SeedVR2_VideoUpscaler.git"),
    ("Nvidia_RTX_Nodes_ComfyUI", "https://github.com/WeiRongHuang/Nvidia_RTX_Nodes_ComfyUI.git"),
]


def check_engine_plugins(no_git):
    print(f"\n{TAG} == 6/8 外部引擎插件（可选，git clone 到 custom_nodes）==")
    if no_git:
        print(f"{TAG} [跳过] --no-git")
        return True
    if not git_available():
        print(f"{TAG} [跳过] 未找到 git，跳过引擎插件 clone（不影响核心功能）")
        return True
    d = custom_nodes_dir()
    all_ok = True
    for name, url in ENGINE_PLUGINS:
        dest = os.path.join(d, name)
        if os.path.isdir(dest):
            print(f"{TAG} [跳过] 已有插件 {name}")
            continue
        print(f"{TAG} [clone] {name} ...")
        r = run(["git", "clone", "--depth", "1", url, dest])
        if r.returncode == 0 and os.path.isdir(dest):
            print(f"{TAG} [完成] {name} -> {dest}")
        else:
            print(f"{TAG} [失败] clone {name}（rc={r.returncode}），可手动:\n"
                  f"         git clone --depth 1 {url} {dest}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# 7) DLSS5 运行时（本地复制 / 官方 release 自动下载）
# ---------------------------------------------------------------------------
DLSS_DLL_NAMES = ("nvngx_dlss.dll", "nvngx_dlssnr.dll", "nvngx.dll_dlssnr.dll")
V2D_RELEASE_BASE = "https://github.com/DaniilSokolyuk/video2dlssnr/releases/download/v1.2"
V2D_LIGHT_ZIP = V2D_RELEASE_BASE + "/video2dlssnr_release_light.zip"   # 225KB, 仅 exe
V2D_FULL_ZIP = V2D_RELEASE_BASE + "/video2dlssnr_release.zip"          # ~247MB, exe + NVIDIA DLL


def find_local_dlss(extra_dirs):
    """在已知位置 + --dlss-dir 里递归找 video2dlssnr.exe / zip / 含 DLL 的目录。"""
    roots = []
    for d in extra_dirs:
        if d:
            roots.append(os.path.abspath(d))
    home = os.path.expanduser("~")
    for cand in ("C:\\BSAI\\DLSS5", os.path.join(home, "DLSS5"),
                 os.path.join(home, "Downloads", "DLSS5")):
        if os.path.isdir(cand):
            roots.append(cand)
    hits = {"exe": None, "zip": None, "dlls": None}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if dirpath[len(root):].count(os.sep) > 4:
                continue
            names = set(files)
            if "video2dlssnr.exe" in names and hits["exe"] is None:
                hits["exe"] = os.path.join(dirpath, "video2dlssnr.exe")
            zips = [fn for fn in files if fn.lower().endswith(".zip") and "video2dlssnr" in fn.lower()]
            if zips and hits["zip"] is None:
                hits["zip"] = os.path.join(dirpath, zips[0])
            if hits["dlls"] is None and all(n in names for n in DLSS_DLL_NAMES):
                hits["dlls"] = dirpath
    return hits


def _extract_zip_into(zip_path, dest):
    """把 zip 内所有 video2dlssnr.exe 与 DLSS DLL 平铺解到 dest（兼容 nested/flat 结构）。"""
    import zipfile
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            rel = m.filename.replace("\\", "/")
            name = rel.rsplit("/", 1)[-1]
            if not name:
                continue
            if name == "video2dlssnr.exe" or name in DLSS_DLL_NAMES:
                with zf.open(m) as s, open(os.path.join(dest, name), "wb") as o:
                    o.write(s.read())
                n += 1
    return n


def check_dlss5(extra_dirs, dlss5_full):
    print(f"\n{TAG} == 7/8 DLSS5 运行时（video2dlssnr + nvngx_*）==")
    dest = os.path.join(models_dir(), "DLSS5")
    os.makedirs(dest, exist_ok=True)
    have_exe = os.path.isfile(os.path.join(dest, "video2dlssnr.exe"))
    have_dlls = all(os.path.isfile(os.path.join(dest, n)) for n in DLSS_DLL_NAMES)
    if have_exe and have_dlls:
        print(f"{TAG} DLSS5 引擎已就绪: {dest}")
        return True
    # 1) 本地已有整合包 -> 复制 / 解包
    hits = find_local_dlss(extra_dirs)
    copied = []
    if hits["dlls"]:
        for n in DLSS_DLL_NAMES:
            src = os.path.join(hits["dlls"], n)
            if os.path.isfile(src) and not os.path.isfile(os.path.join(dest, n)):
                shutil.copy2(src, os.path.join(dest, n))
                copied.append(n)
    if hits["exe"] and not have_exe:
        shutil.copy2(hits["exe"], os.path.join(dest, "video2dlssnr.exe"))
        copied.append("video2dlssnr.exe")
    if copied:
        print(f"{TAG} 从本地整合包复制到 {dest}: {', '.join(copied)}")
    elif hits["zip"]:
        print(f"{TAG} 发现整合包 zip（{hits['zip']}），解包部署 ...")
        try:
            n = _extract_zip_into(hits["zip"], dest)
            print(f"{TAG} 解包完成 -> {dest}（{n} 个文件）")
        except Exception as e:
            print(f"{TAG} [失败] 解包 {hits['zip']}: {e}")
    # 2) 本地复制/解包后复查；仍无 exe 才自动下载官方 release
    have_exe = os.path.isfile(os.path.join(dest, "video2dlssnr.exe"))
    if not have_exe:
        dl_url = V2D_FULL_ZIP if dlss5_full else V2D_LIGHT_ZIP
        print(f"{TAG} 本地无整合包，自动下载官方 release: {os.path.basename(dl_url)} ...")
        tmp = os.path.join(dest, os.path.basename(dl_url) + ".part")
        req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0 BSAI-H3-installer"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            zip_path = tmp[:-len(".part")]
            os.replace(tmp, zip_path)
            n = _extract_zip_into(zip_path, dest)
            os.remove(zip_path)
            print(f"{TAG} 官方 release 下载并解包完成（{n} 个文件）-> {dest}")
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            print(f"{TAG} [失败] 下载官方 release: {e}")
            print(f"{TAG}         可手动: {dl_url}")
    # 3) 复查
    have_exe = os.path.isfile(os.path.join(dest, "video2dlssnr.exe"))
    have_dlls = all(os.path.isfile(os.path.join(dest, n)) for n in DLSS_DLL_NAMES)
    if have_exe and have_dlls:
        print(f"{TAG} DLSS5 引擎已就绪: {dest}")
        return True
    missing = []
    if not have_exe:
        missing.append("video2dlssnr.exe")
    missing += [n for n in DLSS_DLL_NAMES if not os.path.isfile(os.path.join(dest, n))]
    print(f"{TAG} [待办] DLSS5 引擎缺: {', '.join(missing)}")
    print(f"{TAG}         - video2dlssnr.exe: 已自动下载（{V2D_LIGHT_ZIP}）")
    if "nvngx_dlssnr.dll" in missing or "nvngx_dlss.dll" in missing:
        print(f"{TAG}         - nvngx_dlssnr.dll / nvngx_dlss.dll: NVIDIA 专有运行时，需自行获取后放入 {dest}")
        print(f"{TAG}           （来源: 本地 DLSS5 整合包 / NVIDIA 官方 / 用 --dlss5-full 下载含 DLL 的全量包）")
    return False


# ---------------------------------------------------------------------------
# 8) Topaz 引擎（商业，仅检测）
# ---------------------------------------------------------------------------
def check_topaz():
    print(f"\n{TAG} == 8/8 Topaz 引擎（可选，商业软件）==")
    d = os.path.join(models_dir(), "Topaz_Engine")
    legacy = os.path.join(comfyui_root(), "topaz_engine")
    if os.path.isdir(d) and os.listdir(d):
        print(f"{TAG} Topaz 引擎已就绪: {d}")
        return True
    if os.path.isdir(legacy) and os.listdir(legacy):
        print(f"{TAG} Topaz 引擎（旧路径）: {legacy}")
        return True
    print(f"{TAG} [提示] 未检测到 Topaz 商业引擎（可选）。如需 Topaz 生成式档，请自行部署到 {d}")
    return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:]]
    no_pip = "--no-pip" in args
    no_models = "--no-models" in args
    no_git = "--no-git" in args
    dlss5_full = "--dlss5-full" in args
    extra_dirs = []
    i = 0
    while i < len(args):
        if args[i] == "--dlss-dir" and i + 1 < len(args):
            extra_dirs.append(args[i + 1]); i += 2
        elif args[i].startswith("--dlss-dir="):
            extra_dirs.append(args[i].split("=", 1)[1]); i += 1
        else:
            i += 1

    print(f"{TAG} BSAI-H3-upscale-4K 环境自检启动（{sys.executable}）")
    print(f"{TAG} 插件目录: {PLUGIN_DIR}")

    results = {}
    results["1_Python/ComfyUI"] = check_python()
    results["2_PyTorch/CUDA/GPU"] = check_torch()
    results["3_Python包"] = check_pip(no_pip)
    results["4_Real-ESRGAN模型"] = check_esrgan(no_models)
    results["5_人脸修复模型"] = check_face(no_models)
    results["6_引擎插件"] = check_engine_plugins(no_git)
    results["7_DLSS5"] = check_dlss5(extra_dirs, dlss5_full)
    results["8_Topaz(可选)"] = check_topaz()

    print(f"\n{TAG} ============ 汇总 ============")
    for k, v in results.items():
        mark = "✅" if v else "⚠️"
        print(f"{TAG} {mark} {k}")
    print(f"{TAG} ==============================")
    bad = [k for k, v in results.items() if not v]
    if bad:
        print(f"{TAG} 待处理: {', '.join(bad)}（详见上方提示；核心功能=1~4 就绪即可使用 Real-ESRGAN）")
    else:
        print(f"{TAG} 全部就绪！重启 ComfyUI 即可使用。")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

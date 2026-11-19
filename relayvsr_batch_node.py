# -*- coding: utf-8 -*-
"""
RelayVSR Batch Upscale —— BSAI H3 Upscale 4K 插件内置批量子节点
====================================================================

把参考工作流「BSAI功能模块-视频超分放大一键直出2K-4K-8K」的单文件链
(VHS_LoadVideo -> BSAI_H3_Upscale4K[RelayVSR] -> VHS_VideoCombine) 收进一个节点:
扫描输入文件夹内 N 个视频, 逐个同链路超分, 成片写入自定义输出文件夹;
可选 ffmpeg 无损流合并(-c copy)为一个完整视频。

Batch sub-node bundled inside the BSAI H3 Upscale 4K package.
Scans a folder of N videos, runs each through
VHS_LoadVideo -> BSAI_H3_Upscale4K[RelayVSR] -> VHS_VideoCombine,
writes clips to a custom output folder; optional lossless ffmpeg concat (-c copy).

链路复用(不重写引擎):
  1. VHS LoadVideoPath 逐文件读帧 + 音频(沿用源 fps);
  2. BSAI_H3_Upscale4K.upscale(model_name="RelayVSR (one-step Wan VSR)")
     —— 与参考工作流同一节点、同一 RelayVSR 引擎与默认参数;
  3. VHS VideoCombine 按源 fps 编码 h264/yuv420p/crf19 并混入原声;
  4. 成品拷贝到 output_folder/<原名>_RelayVSR_x<scale>.mp4。

依赖(已随主环境安装, 本插件不内嵌模型):
  - ComfyUI-VideoHelperSuite (VHS_LoadVideo / VHS_VideoCombine)
  - 本包自带 BSAI_H3_Upscale4K (RelayVSR 权重)
"""

import os
import re
import sys
import shutil
import subprocess
import importlib.util
import time
import traceback

import folder_paths

try:
    # 包内直接引用主节点(首选, 与主模块同生命周期)
    from .bsai_h3_upscale_4k import BSAI_H3_Upscale4K as _LOCAL_UPSCALER
except Exception:
    _LOCAL_UPSCALER = None


# ---------------------------------------------------------------------------
# 宿主模块定位: 包内引用失败时, 再从 sys.modules / 文件路径兜底。
# ---------------------------------------------------------------------------

def _find_loaded_mod(*substrings, attr=None):
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        low = (getattr(mod, "__name__", "") or "").lower()
        if not low:
            continue
        if any(s in low for s in substrings) and (attr is None or hasattr(mod, attr)):
            return mod
    return None


def _load_file_module(path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_upscaler_class():
    import inspect
    cls = _LOCAL_UPSCALER
    if not (inspect.isclass(cls) and callable(cls) and hasattr(cls, "upscale")):
        cls = None
    if cls is None:
        mod = _find_loaded_mod("upscale_4k", "bsai_h3_upscale", attr="BSAI_H3_Upscale4K")
        cand = getattr(mod, "BSAI_H3_Upscale4K", None) if mod else None
        if inspect.isclass(cand) and callable(cand) and hasattr(cand, "upscale"):
            cls = cand
    if cls is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel in ("BSAI-H3-upscale-4K", "BSAI-H3-upscale-4k", "BSAI-H3-Upscale-4K"):
            p = os.path.join(base, rel, "bsai_h3_upscale_4k.py")
            if os.path.isfile(p):
                m = _load_file_module(p, "bsai_relayvsr_batch_4k")
                c = getattr(m, "BSAI_H3_Upscale4K", None)
                if inspect.isclass(c) and callable(c) and hasattr(c, "upscale"):
                    cls = c
                    break
    if cls is None:
        raise RuntimeError(
            "BSAI RelayVSR Batch: 未找到 BSAI-H3-upscale-4K 插件(BSAI_H3_Upscale4K)。"
            "请确认本插件已安装并启用。"
        )
    return cls


def _is_vhs_class(cls, method):
    """只接受真类 + 可调用 + 带目标方法, 排除同名的 _OpNamespace / 函数等异物。"""
    import inspect
    return inspect.isclass(cls) and callable(cls) and hasattr(cls, method)


def _get_vhs_classes():
    """从已加载模块里找 LoadVideoPath / VideoCombine(ComfyUI 启动时 VHS 已加载);
    校验失败再按文件路径兜底加载。"""
    loader_cls = None
    combine_cls = None
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        low = (getattr(mod, "__name__", "") or "").lower()
        if "videohelpersuite" not in low and "videosuite" not in low:
            continue  # 只在 VHS 模块内找, 避免误命中同名异物
        if loader_cls is None and _is_vhs_class(getattr(mod, "LoadVideoPath", None), "load_video"):
            loader_cls = mod.LoadVideoPath
        if combine_cls is None and _is_vhs_class(getattr(mod, "VideoCombine", None), "combine_video"):
            combine_cls = mod.VideoCombine
        if loader_cls is not None and combine_cls is not None:
            break

    # 文件路径兜底
    if loader_cls is None or combine_cls is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vhs_dir = os.path.join(base, "ComfyUI-VideoHelperSuite", "videohelpersuite")
        try:
            if loader_cls is None:
                m = _load_file_module(os.path.join(vhs_dir, "load_video_nodes.py"), "bsai_rvb_vhs_load")
                if _is_vhs_class(getattr(m, "LoadVideoPath", None), "load_video"):
                    loader_cls = m.LoadVideoPath
            if combine_cls is None:
                m = _load_file_module(os.path.join(vhs_dir, "nodes.py"), "bsai_rvb_vhs_nodes")
                if _is_vhs_class(getattr(m, "VideoCombine", None), "combine_video"):
                    combine_cls = m.VideoCombine
        except Exception:
            pass

    if loader_cls is None or combine_cls is None:
        raise RuntimeError(
            "BSAI RelayVSR Batch: 未找到 ComfyUI-VideoHelperSuite 的 LoadVideoPath / "
            "VideoCombine。请确认 VHS 插件已安装并启用。"
        )
    return loader_cls, combine_cls


def _find_ffmpeg():
    """复用 VHS 解析出的 ffmpeg; 取不到再试 imageio_ffmpeg / 系统 PATH。"""
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        low = (getattr(mod, "__name__", "") or "").lower()
        if "videohelpersuite" in low:
            p = getattr(mod, "ffmpeg_path", None)
            if p and os.path.isfile(p):
                return p
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        p = get_ffmpeg_exe()
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def _concat_list_file(paths, dst):
    list_txt = os.path.join(os.path.dirname(dst), "_bsai_relayvsr_concat.txt")
    with open(list_txt, "w", encoding="utf-8") as fh:
        for p in paths:
            # concat 清单用正斜杠 + 单引号, 规避盘符/空格/中文
            safe = os.path.abspath(p).replace("\\", "/")
            fh.write(f"file '{safe}'\n")
    return list_txt


def _concat_videos(paths, dst, ffmpeg_exe):
    """ffmpeg concat demuxer 无损流合并(-c copy), 不重编码。"""
    list_txt = _concat_list_file(paths, dst)
    try:
        cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
               "-i", list_txt, "-c", "copy", dst]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 60)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[-800:])
    finally:
        try:
            os.remove(list_txt)
        except Exception:
            pass
    return dst


def _concat_videos_reencode(paths, dst, ffmpeg_exe):
    """concat demuxer + 统一重编码(libx264/yuv420p/crf19 + aac),
    兼容混帧率/混参数素材; 非无损, 重新压缩。"""
    list_txt = _concat_list_file(paths, dst)
    try:
        cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", list_txt,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
               "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", dst]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 60 * 4)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[-800:])
    finally:
        try:
            os.remove(list_txt)
        except Exception:
            pass
    return dst


def _clean_path(path):
    """剥掉用户从别处复制带入的引号/首尾空白; 归一化反斜杠。"""
    if path is None:
        return ""
    p = str(path).strip()
    # 成对或单边带入的引号都剥掉(如 "C: / 'D:\x' / D:\x" )
    p = p.strip('"').strip("'").strip()
    if p[:1] in ('"', "'"):
        p = p[1:].strip()
    if p[-1:] in ('"', "'"):
        p = p[:-1].strip()
    return p


def _resolve_dir(path, sub):
    """相对路径按主目录解析: 输入相对 input/, 输出相对 output/。"""
    p = _clean_path(path)
    if not p:
        return None
    # 盘符开头但漏写反斜杠(如 "C:")补成 "C:\"
    m = re.match(r'^([A-Za-z]:)[/\\]?(.*)$', p)
    if m and not os.path.isabs(p):
        p = m.group(1) + "\\" + m.group(2)
    if os.path.isabs(p):
        return os.path.normpath(p)
    root = folder_paths.get_input_directory() if sub == "input" else folder_paths.get_output_directory()
    return os.path.normpath(os.path.join(root, p))


class BSAI_RelayVSR_BatchUpscale:
    """RelayVSR 批量超分: 扫描文件夹 -> 逐文件 LoadVideo -> RelayVSR 超分 -> VideoCombine -> 自定义目录。"""

    EXTS = ("mp4", "mov", "mkv", "webm", "avi", "flv", "ts", "m4v")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入视频文件夹(绝对路径; 相对路径按 input/ 解析)",
                    "tooltip": "批量输入文件夹, 扫描该目录下全部视频(不含子目录), 逐个 RelayVSR 超分。",
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "成品输出文件夹(绝对路径; 相对路径按 output/ 解析)",
                    "tooltip": "自定义输出目录, 不存在则自动创建; 成片命名 <原名>_RelayVSR_x<scale>.mp4。",
                }),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.01,
                                    "tooltip": "超分倍率: 2=1080p档, 4=4K档; RelayVSR 按倍率自动配对 x2/x4 权重。"}),
                "relay_weights": (["Auto (by scale)", "x2 (LightX2V)", "x4"],
                                  {"default": "Auto (by scale)",
                                   "tooltip": "RelayVSR 主干权重: Auto 按 scale 自动选档。"}),
                "relay_detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Flash 解码器细节强度, 1.0=完整模型输出。"}),
                "detail_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05,
                                            "tooltip": "USM 细节增强强度(参考工作流 0.5)。"}),
                "detail_radius": ("FLOAT", {"default": 1.8, "min": 0.3, "max": 8.0, "step": 0.1,
                                          "tooltip": "USM 半径(参考工作流 1.8)。"}),
                "softness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "柔化回 blend, 压过冲(参考工作流 0.1)。"}),
                "temporal_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.8, "step": 0.05,
                                                "tooltip": "光流时域混合; RelayVSR 引擎自带时序, 保持低数值。"}),
                "face_restore": (["Off", "GFPGANv1.4", "CodeFormer", "Small-Face (CodeFormer)"],
                                  {"default": "Off", "tooltip": "人脸修复, 参考工作流为 Off。"}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16,
                                      "tooltip": "0=整图快路径(RTX 30xx+ 推荐); 显存不足再设正分块。"}),
                "batch_frames": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1,
                                        "tooltip": "每批帧数(参考工作流 4)。"}),
                "use_compile": ("BOOLEAN", {"default": True, "tooltip": "torch.compile 固定帧尺寸加速, 首次编译一次。"}),
                "crf": ("INT", {"default": 19, "min": 1, "max": 51, "step": 1,
                                "tooltip": "h264 CRF, 越小越清晰越大(参考工作流 19)。"}),
                "skip_existing": ("BOOLEAN", {"default": True, "tooltip": "输出已存在则跳过, 方便断点续跑。"}),
                "keep_audio": ("BOOLEAN", {"default": True, "tooltip": "保留源视频音频并混入成片。"}),
                "merge_mode": (["无损流合并 (-c copy)", "统一重编码 (Re-encode)", "Off"],
                                {"default": "无损流合并 (-c copy)",
                                 "tooltip": "合并档: 无损流合并=不重编码直拼(要求各片编码一致); 统一重编码=libx264/aac 重编码统一编码参数, 兼容混帧率素材; Off=逐片保存不合并。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("batch_report",)
    FUNCTION = "run_batch"
    CATEGORY = "BSAI/Upscale4K/Batch"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "RelayVSR 批量超分: 输入文件夹 N 个视频逐个走 "
        "VHS_LoadVideo -> BSAI_H3_Upscale4K(RelayVSR) -> VHS_VideoCombine, "
        "成品写入自定义输出文件夹; 单文件失败不中断整批。"
    )

    def run_batch(self, input_folder, output_folder, scale=2.0,
                  relay_weights="Auto (by scale)", relay_detail=1.0,
                  detail_amount=0.5, detail_radius=1.8, softness=0.1,
                  temporal_strength=0.2, face_restore="Off",
                  tile_size=0, batch_frames=4, use_compile=True, crf=19,
                  skip_existing=True, keep_audio=True, merge_mode="无损流合并 (-c copy)"):
        in_dir = _resolve_dir(input_folder, "input")
        out_dir = _resolve_dir(output_folder, "output")
        if not in_dir:
            raise RuntimeError("BSAI RelayVSR Batch: input_folder 为空, 请填写输入视频文件夹路径。")
        if not os.path.isdir(in_dir):
            raise RuntimeError(f"BSAI RelayVSR Batch: 输入文件夹不存在: {in_dir}")
        if not out_dir:
            raise RuntimeError("BSAI RelayVSR Batch: output_folder 为空, 请填写成品输出文件夹路径。")
        os.makedirs(out_dir, exist_ok=True)

        files = []
        for f in sorted(os.listdir(in_dir)):
            full = os.path.join(in_dir, f)
            if not os.path.isfile(full):
                continue
            ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
            if ext in self.EXTS:
                files.append(full)
        if not files:
            raise RuntimeError(f"BSAI RelayVSR Batch: 文件夹内没有视频: {in_dir}")

        upscaler_cls = _get_upscaler_class()
        loader_cls, combine_cls = _get_vhs_classes()
        upscaler = upscaler_cls()
        loader = loader_cls()
        combiner = combine_cls()

        done, skipped, failed = [], [], []
        done_paths = []
        t0 = time.time()

        for i, src in enumerate(files, 1):
            stem = os.path.splitext(os.path.basename(src))[0]
            dst = os.path.join(out_dir, f"{stem}_RelayVSR_x{float(scale):g}.mp4")
            if skip_existing and os.path.exists(dst):
                skipped.append(os.path.basename(dst))
                print(f"[RelayVSR Batch] ({i}/{len(files)}) 已存在, 跳过: {dst}")
                continue

            print(f"[RelayVSR Batch] ({i}/{len(files)}) 开始: {os.path.basename(src)}")
            try:
                # 1. 读帧 + 音频(沿用源 fps)
                images, frame_count, audio, vinfo = loader.load_video(
                    video=_clean_path(src), force_rate=0, custom_width=0, custom_height=0,
                    frame_load_cap=0, skip_first_frames=0, select_every_nth=1,
                    format="AnimateDiff",
                )
                # 2. RelayVSR 超分(与参考工作流同一节点同一引擎)
                out, bw, bh, eff, info = upscaler.upscale(
                    images=images,
                    auto_prefer="Quality",
                    model_name="RelayVSR (one-step Wan VSR)",
                    scale=float(scale),
                    tile_size=int(tile_size),
                    tile_pad=16,
                    batch_frames=int(batch_frames),
                    use_fp16=True,
                    use_compile=bool(use_compile),
                    temporal_strength=float(temporal_strength),
                    detail_amount=float(detail_amount),
                    detail_radius=float(detail_radius),
                    softness=float(softness),
                    face_restore=face_restore,
                    face_det_backend="Auto",
                    face_det_conf=0.15,
                    face_blend=0.7,
                    face_fidelity=0.6,
                    detail_mode="smart",
                    input_adaptive="Auto",
                    face_temporal=0.5,
                    relay_weights=relay_weights,
                    relay_detail=float(relay_detail),
                )
                # 3. 编码 + 混入原声(源 fps)
                fps = float((vinfo or {}).get("source_fps") or 24.0)
                res = combiner.combine_video(
                    frame_rate=fps,
                    loop_count=0,
                    images=out,
                    filename_prefix="BSAI-RelayVSR-Batch/_tmp",
                    format="video/h264-mp4",
                    pingpong=False,
                    save_output=True,
                    audio=audio if keep_audio else None,
                    pix_fmt="yuv420p",
                    crf=int(crf),
                    save_metadata=False,
                    trim_to_audio=False,
                )
                produced = res["result"][0][1] if isinstance(res, dict) else res[0][1]
                if not produced:
                    raise RuntimeError("VideoCombine 未产出文件")
                # 4. 拷入自定义输出目录, 清理 output 临时文件
                shutil.copy2(produced[-1], dst)
                for tmp_file in produced:
                    try:
                        if os.path.isfile(tmp_file) and os.path.abspath(tmp_file) != os.path.abspath(dst):
                            os.remove(tmp_file)
                    except Exception:
                        pass
                done.append(os.path.basename(dst))
                done_paths.append(dst)
                print(f"[RelayVSR Batch] ({i}/{len(files)}) 完成: {os.path.basename(dst)} "
                      f"({frame_count}帧 -> {bw}x{bh}, eff={eff:.2f}x, {info})")
            except Exception as e:
                failed.append(f"{os.path.basename(src)}: {e}")
                print(f"[RelayVSR Batch] ({i}/{len(files)}) 失败, 继续下一个: {e}")
                traceback.print_exc()

        # 5. 可选: 本批成功片合并为一个完整视频(无损流合并 / 统一重编码 / Off)
        merged_path = None
        merge_note = ""
        if merge_mode != "Off" and done_paths:
            if len(done_paths) == 1:
                merge_note = "仅 1 片, 跳过合并(成片即单片)。"
            else:
                ffmpeg_exe = _find_ffmpeg()
                if not ffmpeg_exe:
                    merge_note = "未找到 ffmpeg, 合并跳过(单片已保存)。"
                else:
                    merged_path = os.path.join(
                        out_dir, f"merged_RelayVSR_x{float(scale):g}.mp4")
                    try:
                        if merge_mode.startswith("统一重编码"):
                            _concat_videos_reencode(done_paths, merged_path, ffmpeg_exe)
                            merge_note = "已统一重编码合并(libx264/aac, 非无损)。"
                        else:
                            _concat_videos(done_paths, merged_path, ffmpeg_exe)
                            merge_note = "已无损流合并(-c copy, 不重编码)。"
                        print(f"[RelayVSR Batch] {merge_note} -> {merged_path}")
                    except Exception as e:
                        merge_note = f"合并失败(单片已保存): {e}"
                        merged_path = None
                        print(f"[RelayVSR Batch] {merge_note}")

        elapsed = time.time() - t0
        report = (
            f"RelayVSR Batch 完成: 成功 {len(done)} / 跳过 {len(skipped)} / "
            f"失败 {len(failed)}, 用时 {elapsed/60:.1f} 分钟, 输出目录 {out_dir}"
        )
        if merged_path:
            report += f"\n合并成片: {merged_path}"
        if merge_note:
            report += f"\n合并说明: {merge_note}"
        if failed:
            report += "\n失败清单:\n" + "\n".join(failed)
        print(f"[RelayVSR Batch] {report}")
        if not done and failed:
            raise RuntimeError("BSAI RelayVSR Batch: 全部失败:\n" + "\n".join(failed))
        return (report,)


NODE_CLASS_MAPPINGS = {
    "BSAI_RelayVSR_BatchUpscale": BSAI_RelayVSR_BatchUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_RelayVSR_BatchUpscale": "RelayVSR 批量超分(Batch Upscale)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

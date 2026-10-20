// BSAI H3 upscale 4K — 旧工作流 widget 顺序迁移
//
// v2.9.4 (64e0f6c) 在 dlss_motion 之后插入了 3 个 RTX VFX 参数
// (rtx_vfx_op / rtx_vfx_level / rtx_strength)。LiteGraph 按位置加载
// widgets_values, v2.9.4 之前保存的工作流 (35 个 widget 值) 会从插入点开始
// 整体错位, 导致校验 "Value not in list" 或参数值错乱。
//
// 加载时检测: rtx_vfx_op 的值不是合法选项 (错位时它拿到的是 vosr_cfg 的
// 浮点值), 即判定为旧工作流, 按 v2.9.4 前的参数顺序重排回正确值,
// 插入的 3 个参数复位为默认值。新工作流 (38 值) rtx_vfx_op 必为合法
// 字符串, 直接跳过, 零影响。

import { app } from "../../../scripts/app.js";

// v2.9.4 之前的 widget 顺序 (35 项, 不含 images 输入socket)
const OLD_ORDER = [
    "auto_prefer", "model_name", "scale", "tile_size", "tile_pad",
    "batch_frames", "use_fp16", "use_compile", "temporal_strength",
    "detail_amount", "detail_radius", "softness", "face_restore",
    "face_det_backend", "face_det_conf", "face_blend", "face_fidelity",
    "detail_mode", "input_adaptive", "face_temporal", "dlss_style",
    "dlss_intensity", "dlss_detail", "dlss_motion", "vosr_cfg",
    "vosr_steps", "vosr_scale", "dlss_chain_scale", "rtx_chain_scale",
    "sv2_steps", "sv2_cfg", "sv2_sampler", "sv2_scheduler", "sv2_color",
    "flashvsr_anti_ghost",
];

// v2.9.4 插入参数的默认值 (错位时它们拿着别人的旧值, 必须复位)
const INSERTED_DEFAULTS = {
    rtx_vfx_op: "Denoise",
    rtx_vfx_level: "ULTRA",
    rtx_strength: 1.0,
};

const RTX_OP_VALID = ["Denoise", "Deblur", "去噪", "去模糊"];

app.registerExtension({
    name: "BSAI.H3Upscale4K.LegacyWidgetMigrate",

    loadedGraphNode(node) {
        if (node.comfyClass !== "BSAI_H3_Upscale4K") return;
        const widgets = node.widgets || [];
        const opWidget = widgets.find((w) => w.name === "rtx_vfx_op");
        if (!opWidget) return;
        // 合法选项 = 新工作流 (或 v2.9.4~i18n 间的中文标签工作流), 无需迁移
        if (typeof opWidget.value === "string" && RTX_OP_VALID.includes(opWidget.value)) return;

        // 旧工作流: 前 35 个 widget 当前按位置拿着旧顺序的值, 按名称重排
        const mangled = OLD_ORDER.map((_, k) => (k < widgets.length ? widgets[k].value : undefined));
        for (const wdg of widgets) {
            const k = OLD_ORDER.indexOf(wdg.name);
            if (k >= 0 && mangled[k] !== undefined) wdg.value = mangled[k];
            if (wdg.name in INSERTED_DEFAULTS) wdg.value = INSERTED_DEFAULTS[wdg.name];
        }
        console.log("[BSAI-H3-Upscale4K] migrated pre-v2.9.4 workflow widget order (rtx_vfx params inserted)");
    },
});

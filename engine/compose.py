# -*- coding: utf-8 -*-
"""覆盖渲染：纯色气泡用同色块盖住原文，复杂背景局部 inpaint 或 LaMa 深度学习抹字。

原理：浏览器画面里的日文盖不掉，只能取文字块周围背景色把原文盖住。
- method=cover（默认）：环形采样区颜色纯（纯色气泡）时用同色矩形覆盖；
  颜色杂乱（渐变/图案）时自动回退到局部 cv2.inpaint 修复，再交给 render 回填中文。
- method=cv2：整图 cv2.inpaint（静态图导出可用）。
- method=lama：simple-lama-inpainting（big-lama.pt）整图修复，需 torch。
"""
import hashlib
import json
import os

import cv2
import numpy as np

_LAMA_CACHE = {"key": None, "model": None}  # LaMa 模型缓存（懒加载，只装一次）

#: big-lama.pt（simple-lama-inpainting v0.1.0 发布权重）的 SHA256。
#: 用于识别半截下载/损坏的本地文件；不一致时明确报错，不让 torch 拿着坏文件硬加载。
_LAMA_WEIGHT_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
_LAMA_WEIGHT_NAME = "big-lama.pt"


def sha256_file(path, chunk_size=1 << 20):
    """流式计算文件 SHA256；200MB 权重按块读取，不整文件进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lama_weight(path):
    """校验 LaMa 权重完整性：通过返回 True，文件缺失返回 False，损坏抛 RuntimeError。

    .ok 标记里记录 sha256 与文件大小，只是缓存校验结果、避免每次启动重算哈希；
    大小对不上或标记损坏就重算，哈希仍须与官方发布件一致才放行。
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path + ".ok", encoding="utf-8") as handle:
            record = json.load(handle)
        trusted = (record.get("sha256") == _LAMA_WEIGHT_SHA256
                   and int(record.get("size", -1)) == os.path.getsize(path))
    except (OSError, ValueError, TypeError):
        trusted = False
    if trusted:
        return True
    actual = sha256_file(path)
    if actual != _LAMA_WEIGHT_SHA256:
        raise RuntimeError(
            "LaMa 权重校验失败：文件不完整或不是官方发布件。\n"
            f"路径：{path}\n期望 SHA256：{_LAMA_WEIGHT_SHA256}\n实际 SHA256：{actual}\n"
            "删除该文件后重试，会自动重新下载（约 200MB）。")
    _write_lama_marker(path)
    return True


def _write_lama_marker(path):
    """写 .ok 完成标记；写不进去只影响下次启动多算一遍哈希，不影响使用。"""
    try:
        with open(path + ".ok", "w", encoding="utf-8") as handle:
            json.dump({"sha256": _LAMA_WEIGHT_SHA256, "size": os.path.getsize(path)},
                      handle, ensure_ascii=False)
    except OSError:
        pass


def cover_boxes(image_bgr, boxes, cfg):
    """把每个文字块盖住：按 method 选择抹字方式，返回新的 BGR 图。"""
    # 多列对白合并用于整句翻译/排版；抹字仍使用各原始框，避免擦掉列间背景。
    boxes = [source for box in boxes for source in box.get("source_boxes", [box])]
    inpaint_cfg = cfg.get("inpaint", {})
    method = inpaint_cfg.get("method", "cover")

    if method == "lama":
        return _lama_inpaint(image_bgr, boxes, inpaint_cfg)

    if method == "cv2":
        # 旧的真实 inpaint：仅在需要时保留（静态图导出可用）
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        pad = inpaint_cfg.get("pad", 4)
        h_img, w_img = image_bgr.shape[:2]
        for b in boxes:
            x0, y0 = max(0, b["x"] - pad), max(0, b["y"] - pad)
            x1, y1 = min(w_img, b["x"] + b["w"] + pad), min(h_img, b["y"] + b["h"] + pad)
            mask[y0:y1, x0:x1] = 255
        return cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)

    # 默认 cover：环形区域纯色 -> 同色覆盖；颜色杂乱 -> 局部 inpaint
    gap = inpaint_cfg.get("pad", 3)
    ring_w = inpaint_cfg.get("cover_ring_width", 5)
    flat_std = inpaint_cfg.get("cover_flat_std", 12)
    grain = bool(inpaint_cfg.get("grain", False))
    out = image_bgr.copy()
    h_img, w_img = out.shape[:2]
    for b in boxes:
        color, spread = _sample_surrounding(image_bgr, b, gap=gap, ring_w=ring_w)
        x0, y0 = max(0, b["x"] - gap), max(0, b["y"] - gap)
        x1, y1 = min(w_img, b["x"] + b["w"] + gap), min(h_img, b["y"] + b["h"] + gap)
        if spread > flat_std:
            patch = _inpaint_region(image_bgr, x0, y0, x1, y1, ring_w, grain=grain)
            if patch is not None:
                out[y0:y1, x0:x1] = patch
                continue
        out[y0:y1, x0:x1] = _fill_with_grain(image_bgr, x0, y0, x1, y1, color, ring_w, grain=grain)
    return out


def _grain_std(crop, mask):
    """可选细颗粒估计：避开待抹文字边缘/结构线，以 MAD 抑制离群点。

    随机噪声不是漫画网点或排线的重建，默认关闭。即使手动开启，也不能把
    气泡边线、邻近字迹的高频能量转换成满框麻点；强度上限为 8 灰度级。
    """
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    high = gray - smooth
    # 模糊核会将原字边缘混入外圈，扩大排除区后再取样。
    excluded = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    outside = high[(excluded == 0) & (np.hypot(gx, gy) < 12.0)]
    if outside.size < 16:
        return 0.0
    mad = np.median(np.abs(outside - np.median(outside)))
    return min(8.0, float(1.4826 * mad))


def _add_grain(patch, std, seed=0):
    """给补丁叠加与周围同强度的颗粒噪声；std<=0 时原样返回。

    噪声只在亮度上（三通道同一个值）。逐通道独立加噪会在灰阶漫画上留下
    彩色麻点，比不加还显眼。
    """
    if std <= 0 or patch.size == 0:
        return patch
    noise = np.random.default_rng(seed).normal(0.0, std, patch.shape[:2] + (1,))
    return np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _fill_with_grain(image_bgr, x0, y0, x1, y1, color, margin, grain=False):
    """纯色覆盖；grain=True 时补回周围颗粒，避免盖出一块死平的色块。"""
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), np.uint8)
    patch = np.full((y1 - y0, x1 - x0, 3), color, np.uint8)
    if not grain:
        return patch
    crop, mask, _, _ = _crop_with_mask(image_bgr, x0, y0, x1, y1, margin)
    return _add_grain(patch, _grain_std(crop, mask), seed=_seed(x0, y0))


def _seed(x0, y0):
    """由位置生成固定种子：结果可复现，又不会让所有补丁颗粒完全一样。"""
    return (int(x0) * 73856093) ^ (int(y0) * 19349663)


def _crop_with_mask(image_bgr, x0, y0, x1, y1, margin):
    """裁出比抹除区多 margin 像素的背景块，以及其中“待抹除”区域的掩码。

    返回 (裁剪块, 掩码, 裁剪块左上角在整图中的 x, y)，方便调用方按原坐标取回补丁。
    """
    h, w = image_bgr.shape[:2]
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(w, x1 + margin), min(h, y1 + margin)
    crop = image_bgr[cy0:cy1, cx0:cx1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    mask[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = 255
    return crop, mask, cx0, cy0


def _lama_inpaint(image_bgr, boxes, inpaint_cfg):
    """LaMa 整图抹字：mask = 各文字块外扩 pad，返回修复后的 BGR 图。"""
    pad = inpaint_cfg.get("pad", 3)
    h_img, w_img = image_bgr.shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for b in boxes:
        x0, y0 = max(0, b["x"] - pad), max(0, b["y"] - pad)
        x1, y1 = min(w_img, b["x"] + b["w"] + pad), min(h_img, b["y"] + b["h"] + pad)
        mask[y0:y1, x0:x1] = 255
    if not mask.any():
        return image_bgr.copy()
    model = _get_lama(inpaint_cfg.get("lama_model"))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = model(image_rgb, mask)  # 返回 PIL RGB
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def _get_lama(model_path):
    """加载并缓存 LaMa 模型；加载前先校验权重完整性，防止半截文件。

    model_path 为空时用 simple-lama-inpainting 的默认缓存路径
    （torch hub checkpoints/big-lama.pt），缺失时由该库自动下载，下完再校验。
    """
    key = model_path or ""
    if _LAMA_CACHE["model"] is not None and _LAMA_CACHE["key"] == key:
        return _LAMA_CACHE["model"]
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise RuntimeError(
            "inpaint.method=lama 需要安装 simple-lama-inpainting："
            "pip install simple-lama-inpainting --no-deps；"
            "首次运行会自动下载 big-lama.pt（约 200MB，需要网络）"
        ) from exc
    import torch
    if model_path:
        checkpoint = str(model_path)
        if not os.path.exists(checkpoint):
            raise RuntimeError(
                f"配置的 LaMa 权重不存在：{checkpoint}\n"
                "请修正 config.yaml 的 inpaint.lama_model，或置为 null 走自动下载")
        os.environ["LAMA_MODEL"] = checkpoint  # 该库通过环境变量读取本地权重路径
    else:
        checkpoint = os.path.join(torch.hub.get_dir(), "checkpoints", _LAMA_WEIGHT_NAME)
    verified = verify_lama_weight(checkpoint)
    model = SimpleLama()          # 权重缺失时由库下载到 checkpoint
    if not verified:
        verify_lama_weight(checkpoint)
    _LAMA_CACHE["key"], _LAMA_CACHE["model"] = key, model
    return model


def _inpaint_region(bgr, x0, y0, x1, y1, margin, grain=False):
    """对单个区域做局部 inpaint（只处理裁剪块，比整图快），返回 [y0:y1, x0:x1] 尺寸的修复块。

    裁剪块比抹除区多留 margin 像素背景供 inpaint 参考；若裁剪块没有可用
    背景（贴边过紧）则返回 None，由调用方退回同色覆盖。
    始终把补丁亮度对齐周围背景。grain 默认关闭；仅手动开启时添加轻微随机颗粒，
    不能用它冒充漫画网点/排线重建。复杂背景的结构恢复需另行真页验证。
    """
    if x1 <= x0 or y1 <= y0:
        return None
    crop, mask, cx0, cy0 = _crop_with_mask(bgr, x0, y0, x1, y1, margin)
    outside = crop[mask == 0].reshape(-1, 3)
    if not len(outside):
        return None
    fixed = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
    patch = fixed[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]
    # 亮度对齐独立于可选颗粒开关，关噪点不能同时丢掉颜色修正。
    patch = patch.astype(np.float32) + (np.median(outside, axis=0) - np.median(patch.reshape(-1, 3), axis=0))
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    if not grain:
        return patch
    return _add_grain(patch, _grain_std(crop, mask), seed=_seed(x0, y0))


def _sample_surrounding(bgr, box, gap, ring_w):
    """取文字块外一圈环形区域的中值色（BGR）和颜色离散度（各通道标准差的最大值）。"""
    h, w = bgr.shape[:2]
    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]

    mask = np.zeros((h, w), np.uint8)
    # 外框
    ox0, oy0 = max(0, x - gap - ring_w), max(0, y - gap - ring_w)
    ox1, oy1 = min(w, x + bw + gap + ring_w), min(h, y + bh + gap + ring_w)
    if ox1 > ox0 and oy1 > oy0:
        mask[oy0:oy1, ox0:ox1] = 255
    # 内框（文字块 + gap）挖空
    ix0, iy0 = max(0, x - gap), max(0, y - gap)
    ix1, iy1 = min(w, x + bw + gap), min(h, y + bh + gap)
    if ix1 > ix0 and iy1 > iy0:
        mask[iy0:iy1, ix0:ix1] = 0

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # 环形区为空（文字块几乎占满全图）时，退回整图统计
        px = bgr.reshape(-1, 3)
    else:
        px = bgr[ys, xs]
    color = tuple(int(v) for v in np.median(px, axis=0))  # 已是 BGR 顺序
    spread = float(np.max(px.std(axis=0)))
    return color, spread
# -*- coding: utf-8 -*-
"""OCR：PaddleOCR 常驻实例，检测文字块（坐标 + 原文 + 横竖标记）。

速度说明（本机 32 核 CPU 实测，见 tests/bench_ocr_stage.py、tests/compare_ocr_models.py）：
- paddlepaddle 3.3.1 的 oneDNN 路径一推理就抛 NotImplementedError
  （ConvertPirAttribute2RuntimeAttribute not support ...），所以 enable_mkldnn 只能关。
- 关掉后耗时几乎与图内文字多少无关，是一张 200x100 空图也要 14 秒的固定开销。
  这个开销不在检测尺寸上：实测把 det_limit_side_len 从 960 降到 640 几乎不省时间
  （12.4s vs 12.6s），瓶颈在识别模型本身。
- 换 mobile 档模型收益最大：13.5s -> 2.2s（横排/竖排/密集图识别结果完全一致）。
- textline_orientation 是识别侧唯一明显的省时开关：关掉省约 18%（12.0s -> 9.9s）。
  但它会把个别气泡读成高置信垃圾（绕过 min_score 被当成对白盖上去），故默认保持开。
"""


import math


def _field(res, key, default=None):
    """PaddleOCR 3.x 的结果对象同时支持字典式/属性式访问，这里做兼容。"""
    try:
        v = res[key]
    except Exception:
        v = getattr(res, key, None)
    return v if v is not None else default


#: 识别置信度低于此值的块直接丢弃。实测真实漫画页 0.50 以下全是垃圾、
#: 0.70 以上才是正常对白，取中间的 0.6；想放开就在 config.yaml 调 ocr.min_score。
DEFAULT_MIN_SCORE = 0.6


def _number(value, default):
    """把配置里的值安全转成 float；缺失/非法返回 default。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number      # 排除 nan


def build_ocr_options(cfg):
    """把 config.yaml 的 ocr 段翻成 PaddleOCR 构造参数（纯函数，便于测试）。"""
    ocr_cfg = cfg.get("ocr", {}) or {}
    det_model, rec_model = ocr_cfg.get("det_model"), ocr_cfg.get("rec_model")
    options = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": bool(ocr_cfg.get("textline_orientation", True)),
        "text_det_limit_side_len": int(ocr_cfg.get("det_limit_side_len", 960)),
        # 本机 paddlepaddle 3.3.1 的 oneDNN 推理必崩，只能关掉；实测关掉后仍靠 mobile 档拿回速度。
        "enable_mkldnn": False,
    }
    if det_model or rec_model:
        # 指定了模型名时 PaddleOCR 会忽略 lang，索性不传，免得每次都打警告。
        if det_model:
            options["text_detection_model_name"] = det_model
        if rec_model:
            options["text_recognition_model_name"] = rec_model
    else:
        options["lang"] = ocr_cfg.get("lang", "japan")
    return options


def _glyph_size(box):
    """估一个竖排框的"字号"（px）。

    取 sqrt(w × h ÷ 字数)。框宽 = 列数 × 字号，框高 ≈ 该框最长列的字符数 × 字号，
    两个未知量相乘后恰好抵消，所以不管"一个框里塞了 2~3 列"还是"各列字数不等"，
    同一气泡算出来的值都一致。

    实测（真实漫画页）：同一气泡的列 28.6/28.8、30.2/32.2、23.6/23.0；
    而不同气泡之间差得远。之前用「框宽比值 ≤1.5」判定是错的——PaddleOCR
    常把 2~3 列并进同一个框，宽窄天然差一倍，于是真实气泡的列全被拒了。
    """
    n = max(1, len(box.get("text") or ""))
    return math.sqrt(max(1.0, float(box.get("w", 1)) * float(box.get("h", 1))) / n)


def merge_vertical_columns(boxes):
    """合并紧邻的日文竖排列，右列优先；不把半句话分别送去翻译。

    判据（同气泡的列必然同时满足）：
    1. 都是明显竖排（高 ≥ 2×宽）、字数 ≥ 2；
    2. **字号接近**（_glyph_size 比值 ≤ 1.4）——比框宽可靠得多；
    3. 横向紧邻：列间距在 -0.5 ~ +0.6 倍框宽之间（真实检测框常轻微重叠）；
    4. 纵向大幅重叠（≥ 较矮列的 60%）；
    5. **顶对齐 / 底对齐 / 居中对齐 三者至少一种吻合**（偏差 ≤ 0.8 字号）。
       真实检测框多数顶对齐，末列短时会居中；而错开的阶梯状排列三种都不吻合，
       于是不会经中间列级联把两段对白并成一大团。

    输出按 (上, 左) 排序，呈现顺序稳定。
    """
    def adjacent(a, b):
        if not all(box.get("vertical") and box["h"] > box["w"]
                   and len(box["text"]) >= 2 and box["w"] > 0 for box in (a, b)):
            return False
        # 长宽比也不能用框宽判断：一个框可能含 2~3 列（w = 列数 × 字号），
        # 于是"高 ≥ 2×宽"会把正常的多列框误判成横排。改用字号——一列至少
        # 堆到两个字才算竖排文字行。
        if any(box["h"] < _glyph_size(box) * 1.5 for box in (a, b)):
            return False
        glyph_a, glyph_b = _glyph_size(a), _glyph_size(b)
        if max(glyph_a, glyph_b) > min(glyph_a, glyph_b) * 1.4:
            return False
        left, right = sorted((a, b), key=lambda box: box["x"])
        gap = right["x"] - left["x"] - left["w"]
        widest = max(a["w"], b["w"])
        # 真实检测框贴合字身，相邻竖列常常横向轻微重叠（实测 -4px），
        # 所以允许最多半个框宽的负间距；重叠过多则更像同一列被检出两次，不合并。
        if gap > widest * 0.6 or gap < -widest * 0.5:
            return False
        overlap = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
        if overlap < min(a["h"], b["h"]) * 0.6:
            return False
        top = abs(a["y"] - b["y"])
        bottom = abs((a["y"] + a["h"]) - (b["y"] + b["h"]))
        centre = abs((a["y"] + a["h"] / 2) - (b["y"] + b["h"] / 2))
        return min(top, bottom, centre) <= max(glyph_a, glyph_b) * 0.8

    def reachable(box, group):
        """box 能否并入 group：至少与组内一列相邻，且与整组外框对得上。"""
        if not any(adjacent(box, member) for member in group):
            return False
        top = min(member["y"] for member in group)
        bottom = max(member["y"] + member["h"] for member in group)
        centre = (top + bottom) / 2
        glyph = _glyph_size(box)
        return min(abs(box["y"] - top),
                   abs(box["y"] + box["h"] - bottom),
                   abs(box["y"] + box["h"] / 2 - centre)) <= glyph * 0.8

    groups = []
    for box in sorted(boxes, key=lambda item: item["x"], reverse=True):
        for group in groups:
            if reachable(box, group):
                group.append(box)
                break
        else:
            groups.append([box])

    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(dict(group[0]))
            continue
        # 按列中心而不是左沿排：检测框会横向重叠（实测 -8px），左沿顺序可能把
        # 左右两列颠倒，读出来就是倒句。
        columns = sorted(group, key=lambda box: box["x"] + box["w"] / 2, reverse=True)
        x, y = min(b["x"] for b in group), min(b["y"] for b in group)
        right = max(b["x"] + b["w"] for b in group)
        bottom = max(b["y"] + b["h"] for b in group)
        chars = sum(len(b["text"]) for b in group)
        merged.append({"x": x, "y": y, "w": right - x, "h": bottom - y,
                       "text": "".join(b["text"] for b in columns),
                       "score": sum(b["score"] * len(b["text"]) for b in group) / chars,
                       "vertical": True, "source_boxes": [dict(b) for b in columns]})
    merged.sort(key=lambda box: (box["y"], box["x"]))
    return merged


def filter_low_confidence(boxes, min_score):
    """丢掉识别置信度过低的文字块。

    真实漫画页实测：置信度 0.11~0.50 全是垃圾（"n"、"D·"、"S-我へi"、"中"、"□"），
    0.70 以上才是正常对白，中间有干净分界。低置信块送去翻译只会拿回同样一句
    垃圾并盖到漫画上（模型甚至可能回一句"无法辨认"），不如留着原文不抹。
    min_score <= 0 或非法值表示不过滤。
    """
    try:
        threshold = float(min_score)
    except (TypeError, ValueError):
        return list(boxes)
    if threshold != threshold:        # nan：比较恒为 False，会把所有块都丢掉
        return list(boxes)
    if threshold <= 0:
        return list(boxes)
    kept = []
    for box in boxes:
        try:
            score = float(box.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        if score >= threshold:
            kept.append(box)
    return kept


class OcrEngine:
    """持有单个 PaddleOCR 实例，抓屏循环/多次翻译时复用，避免每次重载模型。"""

    def __init__(self, cfg):
        from paddleocr import PaddleOCR
        ocr_cfg = cfg.get("ocr", {})
        self._merge_columns = ocr_cfg.get("merge_vertical_columns", True)
        self.min_score = _number(ocr_cfg.get("min_score"), DEFAULT_MIN_SCORE)
        self._ocr = PaddleOCR(**build_ocr_options(cfg))

    def detect(self, image_bgr):
        """检测文字块（横排 / 竖排），返回 [{"x","y","w","h","text","score","vertical"}, ...]。"""
        result = self._ocr.predict(image_bgr)
        boxes = []
        for res in (result or []):
            texts = _field(res, "rec_texts", []) or []
            scores = _field(res, "rec_scores", []) or []
            polys = _field(res, "dt_polys", None)
            # 检测框数量对不上识别文本时，回退到识别框
            if polys is None or len(polys) != len(texts):
                polys = _field(res, "rec_polys", None)
            if polys is None:
                continue
            for poly, text, score in zip(polys, texts, scores):
                if not text:
                    continue
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - x), int(max(ys) - y)
                boxes.append({"x": x, "y": y, "w": w, "h": h,
                              "text": str(text), "score": float(score),
                              "vertical": h > w})  # 高 > 宽视为竖排
        return merge_vertical_columns(boxes) if self._merge_columns else boxes
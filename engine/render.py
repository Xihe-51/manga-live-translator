# -*- coding: utf-8 -*-
"""排版回填：自适应字号、中文断行、居中绘制（横排 / 竖排）。

可读性靠两件事兜底：
- auto_color：逐块看背景明暗决定用深字还是浅字。固定黑字压在深色画面上
  就是一团糊，这是漫画嵌字最常见的问题。
- stroke_ratio：给字加描边，描边色自动取字色的反色。纯色气泡上反色描边
  与底色一致、看不出来；背景一复杂就靠它把字从画面里"抠"出来。
"""
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageStat

DARK_TEXT, LIGHT_TEXT = "#101010", "#f5f5f5"


def _number(value, default):
    """把配置里的值安全转成 float；缺失/非法返回 default。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number      # 排除 nan


def wrap_text(text, font, max_w, draw):
    """中文逐字贪心断行。"""
    lines = []
    line = ""
    for ch in text:
        test = line + ch
        if draw.textlength(test, font=font) <= max_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = ch
    if line:
        lines.append(line)
    return lines


def fit_font_size(text, box_w, box_h, font_path, lo=8, hi=200, line_gap=1.2):
    """二分查找能放进矩形内的最大字号（横排）。"""
    best = lo
    tmp = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(tmp)
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        asc, desc = font.getmetrics()
        line_h = (asc + desc) * line_gap
        lines = wrap_text(text, font, max(1, box_w), draw)
        if line_h * len(lines) <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def fit_font_size_vertical(text, box_w, box_h, font_path, lo=8, hi=200, line_gap=1.2):
    """二分查找能放进矩形内的最大字号（竖排）。

    竖排规则：逐字纵排、多列从右到左；竖向步进与列间距都取 字号 × line_gap。
    每列字数 = floor(box_h / 步进)，列数 = ceil(字数 / 每列字数)，要求 列数×步进 ≤ box_w。
    """
    n = len(text)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        step = mid * line_gap
        per_col = int(box_h // step)
        if per_col < 1:
            hi = mid - 1
            continue
        cols = (n + per_col - 1) // per_col
        if cols * step <= box_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def draw_vertical(draw, text, box, font, size, line_gap, color="black", stroke=0, stroke_fill=None):
    """竖排绘制：右→左，共用顶部基线；按实际字形居中而非字体 ascender。"""
    if not text:
        return
    step = size * line_gap
    per_col = max(1, int(box["h"] // step))
    chunks = [text[i:i + per_col] for i in range(0, len(text), per_col)]
    total_w = len(chunks) * step
    x0 = box["x"] + (box["w"] - total_w) / 2
    # 末列较短时也从第一列顶端起笔，不能独立居中导致读序像碎句。
    ty = box["y"] + (box["h"] - max(map(len, chunks)) * step) / 2
    for i, chunk in enumerate(chunks):
        col_x = x0 + (len(chunks) - 1 - i) * step
        for j, ch in enumerate(chunk):
            left, top, right, bottom = draw.textbbox((0, 0), ch, font=font,
                                                     stroke_width=stroke)
            draw.text((col_x + (step - (right - left)) / 2 - left,
                       ty + j * step + (step - (bottom - top)) / 2 - top),
                      ch, font=font, fill=color, stroke_width=stroke, stroke_fill=stroke_fill)


def luminance(color):
    """颜色的相对亮度（0~1）；解析失败按最暗处理。"""
    try:
        r, g, b = ImageColor.getrgb(color)[:3]
    except (ValueError, TypeError):
        return 0.0
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def text_style(cfg):
    """读取译文样式。

    font_scale 限制在 0.5~2.0；color 是固定字色；auto_color=True 时改为逐块
    按背景明暗选深/浅字色；stroke_ratio 是描边宽度占字号的比例，0 表示不描边。
    """
    render_cfg = cfg.get("render", {}) or {}
    scale = _number(render_cfg.get("font_scale"), 1.0)
    if scale <= 0:
        scale = 1.0
    stroke = _number(render_cfg.get("stroke_ratio"), 0.08)
    return {
        "scale": min(2.0, max(0.5, scale)),
        "color": render_cfg.get("color") or "black",
        "auto_color": bool(render_cfg.get("auto_color", True)),
        "stroke_ratio": min(0.5, max(0.0, stroke)),
    }


def background_luminance(image_pil, box):
    """文字块所在背景的中位亮度（0~255）。

    用中位数而不是均值：抹字没抹干净时残留的原文不会把结果带偏。
    """
    x0, y0 = max(0, box["x"]), max(0, box["y"])
    x1, y1 = min(image_pil.width, box["x"] + box["w"]), min(image_pil.height, box["y"] + box["h"])
    if x1 <= x0 or y1 <= y0:
        return 255.0
    return float(ImageStat.Stat(image_pil.crop((x0, y0, x1, y1)).convert("L")).median[0])


def pick_text_color(brightness, style):
    """按背景亮度选字色：亮底配深字，暗底配浅字。"""
    if not style["auto_color"]:
        return style["color"]
    return DARK_TEXT if brightness >= 128 else LIGHT_TEXT


def stroke_for(color, ratio, size):
    """返回 (描边宽度, 描边色)；描边色取字色反色，好让纯色背景上自然隐形。"""
    if ratio <= 0:
        return 0, None
    width = max(1, int(round(size * ratio)))
    return width, (DARK_TEXT if luminance(color) >= 0.5 else LIGHT_TEXT)


def render_texts(image_pil, boxes, translations, cfg):
    """把中文按原文字块位置回填到图像上（原地修改，返回同一 PIL 图）。

    横排块按原逻辑断行排版；vertical=True 的块逐字纵排。
    字色与描边逐块决定，所以先把所有背景亮度采样完再落笔，避免后画的块
    采样到自己刚写上去的字。
    """
    render_cfg = cfg.get("render", {})
    font_path = render_cfg.get("font_path", "C:/Windows/Fonts/msyh.ttc")
    min_font = render_cfg.get("min_font_size", 10)
    line_gap = render_cfg.get("line_gap", 1.2)
    style = text_style(cfg)

    pending = [b for b in boxes if b["text"] in translations]
    brightness = [background_luminance(image_pil, b) for b in pending]

    draw = ImageDraw.Draw(image_pil)
    for b, bright in zip(pending, brightness):
        zh = translations[b["text"]]
        color = pick_text_color(bright, style)

        if b.get("vertical"):
            fitted = fit_font_size_vertical(zh, b["w"], b["h"], font_path, min_font, 200, line_gap)
            size = max(1, round(fitted * style["scale"]))
            font = ImageFont.truetype(font_path, size)
            stroke, stroke_fill = stroke_for(color, style["stroke_ratio"], size)
            draw_vertical(draw, zh, b, font, size, line_gap, color, stroke, stroke_fill)
            continue

        fitted = fit_font_size(zh, b["w"], b["h"], font_path, min_font, 200, line_gap)
        size = max(1, round(fitted * style["scale"]))
        font = ImageFont.truetype(font_path, size)
        lines = wrap_text(zh, font, b["w"], draw)
        stroke, stroke_fill = stroke_for(color, style["stroke_ratio"], size)

        asc, desc = font.getmetrics()
        line_h = int((asc + desc) * line_gap)
        total_h = line_h * len(lines)

        ty = b["y"] + (b["h"] - total_h) / 2          # 垂直居中
        for i, ln in enumerate(lines):
            w = draw.textlength(ln, font=font)
            tx = b["x"] + (b["w"] - w) / 2             # 水平居中
            draw.text((tx, ty + i * line_h), ln, font=font, fill=color,
                      stroke_width=stroke, stroke_fill=stroke_fill)
    return image_pil
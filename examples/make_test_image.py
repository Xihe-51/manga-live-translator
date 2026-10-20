# -*- coding: utf-8 -*-
"""生成测试图：test.png（3 句横排）+ test_vertical.png（竖排 / 多列竖排）。"""
import os

from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/YuGothR.ttc"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_bubble(draw, x, y, bw, bh):
    draw.rounded_rectangle([x, y, x + bw, y + bh], radius=20, fill=(255, 255, 255))


def draw_vertical_text(draw, text, x, y, bw, bh, font, step):
    """在气泡里逐字纵排（多列从右到左），用于生成竖排测试图。"""
    per_col = max(1, int(bh // step))
    chunks = [text[i:i + per_col] for i in range(0, len(text), per_col)]
    total_w = len(chunks) * step
    x0 = x + (bw - total_w) / 2
    for i, chunk in enumerate(chunks):
        col_x = x0 + (len(chunks) - 1 - i) * step      # 第一列在最右
        ty = y + (bh - len(chunk) * step) / 2
        for j, ch in enumerate(chunk):
            w = draw.textlength(ch, font=font)
            draw.text((col_x + (step - w) / 2, ty + j * step), ch, font=font, fill=(0, 0, 0))


def make_horizontal(path):
    W, H = 900, 500
    img = Image.new("RGB", (W, H), (232, 232, 232))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 40)
    # (文本, 气泡左上x, 气泡左上y, 气泡宽, 气泡高)
    items = [
        ("おはよう",       80,  80, 320, 120),
        ("今日はいい天気", 80, 240, 420, 120),
        ("私は漫画が好き", 500, 80, 340, 120),
    ]
    for text, x, y, bw, bh in items:
        draw_bubble(draw, x, y, bw, bh)
        draw.text((x + bw / 2, y + bh / 2), text, font=font, fill=(0, 0, 0), anchor="mm")
    img.save(path)
    print(f"saved {path}")


def make_vertical(path):
    W, H = 900, 500
    img = Image.new("RGB", (W, H), (232, 232, 232))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 40)
    # 气泡均为高 > 宽，触发竖排分支；(文本, x, y, 宽, 高)
    items = [
        ("おやすみなさい", 90, 60, 140, 380),
        ("今日はとても楽しい一日", 420, 60, 260, 380),
    ]
    for text, x, y, bw, bh in items:
        draw_bubble(draw, x, y, bw, bh)
        draw_vertical_text(draw, text, x, y, bw, bh, font, 48)
    img.save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    make_horizontal(os.path.join(OUT_DIR, "test.png"))
    make_vertical(os.path.join(OUT_DIR, "test_vertical.png"))
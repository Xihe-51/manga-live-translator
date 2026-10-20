# -*- coding: utf-8 -*-
"""译文对照导出：把 {原文: 译文} 渲染成便于校对的纯文本。

设计取向是「帮人扫一眼看出哪里译错」，所以：
  - 原文一行、译文一行，中间不留行内标记——眼睛上下扫比左右对齐更快；
  - 段与段之间空一行，避免连续多句糊成一大坨；
  - 纯符号碎片（OCR 把标点、边框线当成文字）默认丢掉，它们只会干扰阅读；
  - 保留首次出现顺序，而不是按字典排序——顺序乱了就不再对应漫画阅读节奏。

这里只做「字符串 → 字符串」，不碰文件系统，方便单独测试。
"""
from __future__ import annotations

#: 只有这些字符的块当噪声丢掉（标点、空白、常见装饰符号）。
_NOISE_CHARS = set(" \t\r\n　.,;:!?'\"`~^_=+-*/\\|<>()[]{}<>"
                   "、。，；：！？…—－·•「」『』【】〈〉《》〈〉“”‘’"
                   "♪☆★○●◎◇◆□■△▲※→←↑↓")


def is_noise(text):
    """判断一段文字是否只是标点/符号碎片，没有实际内容。"""
    stripped = str(text or "").strip()
    if not stripped:
        return True
    return all(ch in _NOISE_CHARS for ch in stripped)


def clean_pairs(pairs, drop_noise=True):
    """把各种形态的译文数据规整成 [(原文, 译文), ...]，保持首次出现顺序。

    接受 dict（{原文: 译文}）或 list（[(原文, 译文)] / [{"source":..., "target":...}]），
    因为 pipeline 与旧网页面板两条路径给过来的结构不完全一致。
    """
    items = []
    if isinstance(pairs, dict):
        items = list(pairs.items())
    elif isinstance(pairs, (list, tuple)):
        for entry in pairs:
            if isinstance(entry, dict):
                source = entry.get("source", entry.get("src", ""))
                target = entry.get("translation",
                                   entry.get("target", entry.get("dst", "")))
                items.append((source, target))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                items.append((entry[0], entry[1]))

    seen = set()
    cleaned = []
    for source, target in items:
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target:
            continue
        if drop_noise and (is_noise(source) or is_noise(target)):
            continue
        if source in seen:
            # 同一句在同一页重复出现只留一次，避免同一对白刷屏。
            continue
        seen.add(source)
        cleaned.append((source, target))
    return cleaned


#: 标记行前缀；统一成一处，改格式只动这里。
_MARK_PREFIX = "[!] 标记："


def mark_reason(marks, source):
    """取某句的标记理由；marks 支持 {原文: 理由} 或 set/list（理由用默认值）。

    阅读时标记只记「译得不好」，导出时才需要理由；这里两种形态都接住，
    方便以后加「漏译」「人称不一致」等分类而不改调用方。
    """
    if not marks:
        return ""
    if isinstance(marks, dict):
        reason = marks.get(source)
        return str(reason) if reason else ""
    if source in marks:
        return "译得不好"
    return ""


def render_pairs(pairs, header=None, drop_noise=True, marks=None):
    """渲染成对照文本；没有有效内容时返回空串（调用方据此判断「没东西可导出」）。

    marks 里被标记的句子会在译文下方追一行「[!] 标记：…」，扫描时一眼可见；
    表头顺带给出标记总数，方便事后知道这批译文要复核多少句。
    """
    cleaned = clean_pairs(pairs, drop_noise=drop_noise)
    if not cleaned:
        return ""
    blocks = []
    marked = 0
    for source, target in cleaned:
        block = f"{source}\n{target}"
        reason = mark_reason(marks, source)
        if reason:
            marked += 1
            block += f"\n  {_MARK_PREFIX}{reason}"
        blocks.append(block)
    body = "\n\n".join(blocks)
    if header:
        head = str(header)
        if marked:
            head += f"（{marked} 句已标记待复核）"
        return f"{head}\n\n{body}\n"
    return body + "\n"


def render_pages(pages, title=None, marks=None):
    """渲染多页导出。pages 是 [(页码或来源说明, pairs), ...]。

    没有有效内容的页会被跳过；全部为空时返回空串。marks 按原文匹配，
    跨页生效——同一句在多页出现时都会带上标记。
    """
    sections = []
    index = 0
    for label, pairs in pages:
        index += 1
        body = render_pairs(pairs, marks=marks)
        if not body:
            continue
        heading = f"===== {label} =====" if label else f"===== 第 {index} 页 ====="
        sections.append(heading + "\n\n" + body.rstrip("\n"))
    if not sections:
        return ""
    text = "\n\n".join(sections) + "\n"
    if title:
        return f"{title}\n\n{text}"
    return text


def default_export_name(prefix="译文对照", when=None, folder=None, suffix=0):
    """给导出文件起个带时间戳的名字，避免多次导出互相覆盖。

    时间戳只精确到秒，同一秒内导出两次会撞名；传入 folder 时会检查是否已存在，
    并在必要时追加 _2、_3… 保证不覆盖已有文件。
    """
    import datetime
    import os
    moment = when or datetime.datetime.now()
    stem = f"{prefix}_{moment.strftime('%Y%m%d_%H%M%S')}"
    if suffix:
        stem += f"_{suffix}"
    name = f"{stem}.txt"
    if folder is None:
        return name
    candidate = name
    index = 1
    while os.path.exists(os.path.join(folder, candidate)):
        index += 1
        candidate = f"{stem}_{index}.txt"
    return candidate

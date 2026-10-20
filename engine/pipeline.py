# -*- coding: utf-8 -*-
"""单页处理编排：OCR → 覆盖 → 翻译 → 排版回填。"""
import time

import cv2
from PIL import Image

from .ocr import OcrEngine, filter_low_confidence
from .translate import (Translator, detect_language, is_garbled_translation,
                        is_untranslatable_noise, language_name)
from .compose import cover_boxes
from .render import render_texts


class Pipeline:
    """持有常驻的 OCR / 翻译器，重复处理时避免重载模型。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ocr = OcrEngine(cfg)
        self.translator = Translator(cfg)

    def recognize_only(self, image_bgr, progress=None):
        """只识别不翻译（GUI 诊断模式）；不抹字、不翻译，只在图上标注编号与框。"""
        return _recognize_only(self, image_bgr, progress)

    def recognize(self, image_bgr, progress=None):
        """只跑 OCR，返回 (文字块, 耗时秒)。翻译链路与诊断链路共用同一套识别。"""
        notify = progress or (lambda message: None)
        notify("正在识别文字（OCR）…")
        t0 = time.perf_counter()
        raw_boxes = self.ocr.detect(image_bgr)
        boxes = filter_low_confidence(raw_boxes, getattr(self.ocr, "min_score", 0))
        elapsed = time.perf_counter() - t0
        notify(f"OCR 完成：检测到 {len(boxes)} 个文字块，用时 {elapsed:.1f} 秒")
        dropped = len(raw_boxes) - len(boxes)
        if dropped:
            notify(f"丢弃 {dropped} 个置信度过低的文字块（保留原文不抹字），"
                   f"阈值 ocr.min_score={getattr(self.ocr, 'min_score', 0)}")
        for i, b in enumerate(boxes, 1):
            tag = "竖排" if b.get("vertical") else "横排"
            print(f'  [{i}] ({b["x"]},{b["y"]},{b["w"]},{b["h"]}) {tag} "{b["text"]}" 置信度 {b["score"]:.2f}')
        return boxes, elapsed

    def process(self, image_bgr, progress=None):
        """处理一页；可选进度回调供 GUI 显示，CLI 调用保持兼容。"""
        notify = progress or (lambda message: None)
        boxes, t_ocr = self.recognize(image_bgr, progress=notify)

        if boxes:
            override = str(self.cfg.get("llm", {}).get("source_lang", "auto") or "auto").strip().lower()
            fixed_language = override in ("ja", "en", "ko", "ru")
            language = override if fixed_language else detect_language([b["text"] for b in boxes])
            if language == "zh":
                # 已经是中文：不抹字也不翻译，原图返回，避免把选区内容擦成白块。
                notify(f"自动识别：选区内 {len(boxes)} 个文字块已经是中文，无需翻译")
                return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), [], {}
            if language != "ja":
                notify(f"自动识别原文语言：{language_name(language)}，将翻译成简体中文")
            # 选区里常混着界面文字（文件管理器、聊天窗）。整段判定会漏掉这种情况，
            # 于是中文/纯数字块被塞进外文提示词，模型回一句"请提供英文对白"并盖掉原文。
            # 逐块再筛一遍：已是中文或没有文字系统（纯数字/符号）的一律保持原样。
            # 固定日文时纯汉字可能是日文对白/人名，不能继续被“已是中文”过滤。
            # 自动模式保留混合界面保护；纯符号始终不送给模型。
            # 日文页另有一道硬过滤：既无假名也无汉字的碎片是 OCR 噪声，不是对白——
            # 实测拟声词「グッ」被读成 "$\\r$"（conf 0.63，越过 min_score），
            # 模型把它译成"1."盖上去，真实内容被抹掉换成垃圾，比不翻更糟。
            keep = [b for b in boxes if detect_language([b["text"]]) != "unknown"
                    and (fixed_language or detect_language([b["text"]]) != "zh")
                    and not (language == "ja" and is_untranslatable_noise(b["text"]))]
            if len(keep) != len(boxes):
                skipped = "纯符号或无日文内容" if fixed_language else "已是中文或纯符号"
                notify(f"跳过 {len(boxes) - len(keep)} 个{skipped}的文字块，保持原样")
            if fixed_language:
                notify(f"使用固定原文语言：{language_name(language)}")
            elif language == "ja":
                notify("自动模式无法区分纯汉字日文与中文；阅读日漫可在设置页固定原文为日文")
            boxes = keep
            if not boxes:
                return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), [], {}

        texts = [b["text"] for b in boxes]
        t2 = time.perf_counter()
        translations = self.translator.translate(texts, progress=notify) if texts else {}
        t_trans = time.perf_counter() - t2
        for src, dst in translations.items():
            print(f'  翻译 "{src}" -> "{dst}"')

        # 只有拿到可用译文的框才抹字：翻译失败（空）与乱码/退化输出（U+FFFD、
        # 高频重复字符等）一律保持原样，避免把气泡擦成空白或盖上乱码。
        # 被丢弃的乱码译文留样到 rejects，方便事后回看模型吐了什么。
        usable = set()
        for text, target in translations.items():
            if not target:
                continue
            if is_garbled_translation(target):
                self.translator.rejects.record("乱码或退化输出", text, target)
                continue
            usable.add(text)
        dropped = sum(1 for b in boxes if b["text"] not in usable)
        boxes = [b for b in boxes if b["text"] in usable]
        if dropped:
            notify(f"有 {dropped} 个文字块没有可用的译文（失败或疑似乱码），保持原样")
        notify("正在清除原文…")
        t1 = time.perf_counter()
        covered = cover_boxes(image_bgr, boxes, self.cfg)
        t_cover = time.perf_counter() - t1

        notify("正在排版回填译文…")
        t3 = time.perf_counter()
        image_pil = Image.fromarray(cv2.cvtColor(covered, cv2.COLOR_BGR2RGB))
        image_pil = render_texts(image_pil, boxes, translations, self.cfg)
        t_render = time.perf_counter() - t3

        print(f"  用时：OCR {t_ocr:.2f}s | 抹字 {t_cover:.2f}s | 翻译 {t_trans:.2f}s | 回填 {t_render:.2f}s", flush=True)
        return image_pil, boxes, translations


def recognize_image(image_bgr, cfg, progress=None):
    """只识别不翻译：返回 (标注图 PIL, boxes, language)。

    诊断专用——先看 OCR 拿到什么，才能区分"识别错"和"翻译错"。
    不抹字、不翻译、不改变原图内容，只在每块左上角画一个标号和框。
    """
    notify = progress or (lambda message: None)
    pipeline = Pipeline(cfg)
    boxes, _ = pipeline.recognize(image_bgr, progress=notify)
    language = detect_language([b["text"] for b in boxes]) if boxes else "unknown"
    notify(f"识别到 {len(boxes)} 个文字块，判定原文语言：{language_name(language)}")
    annotated = _annotate_boxes(image_bgr, boxes)
    return Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)), boxes, language


def _recognize_only(pipeline, image_bgr, progress):
    """Pipeline 实例方法版本，供常驻 Pipeline 复用（GUI 诊断模式用）。"""
    notify = progress or (lambda message: None)
    boxes, _ = pipeline.recognize(image_bgr, progress=notify)
    language = detect_language([b["text"] for b in boxes]) if boxes else "unknown"
    notify(f"识别到 {len(boxes)} 个文字块，判定原文语言：{language_name(language)}")
    annotated = _annotate_boxes(image_bgr, boxes)
    return Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)), boxes, language


def _annotate_boxes(image_bgr, boxes):
    """在识别框上画编号与边框，便于人工核对逐块识别结果。"""
    annotated = image_bgr.copy()
    for i, b in enumerate(boxes, 1):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 128, 0), 1)
        cv2.putText(annotated, str(i), (x, max(10, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return annotated


def process_image(image_bgr, cfg, out=None):
    """一次性处理（内部建 Pipeline）。返回 (中文图 PIL, boxes, translations)。"""
    pipeline = Pipeline(cfg)
    image_pil, boxes, translations = pipeline.process(image_bgr)
    if out:
        image_pil.save(out)
    return image_pil, boxes, translations
# -*- coding: utf-8 -*-
"""无需真实截屏、OCR 或网络的回归测试。运行：python -m unittest discover -s tests -v"""
import base64
import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPoint, QRect, Qt, Slot
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWebChannel import QWebChannel, QWebChannelAbstractTransport
from PySide6.QtWidgets import QApplication
from PIL import Image, ImageDraw
import numpy as np
import cv2

from ui.bridge import Bridge, parse_region
from ui.region_selector import RegionSelector
from engine.ocr import build_ocr_options, filter_low_confidence
from engine.translate import (Translator, build_models_url, detect_language, fetch_models,
                              has_latin_letters, is_garbled_translation, is_meta_response,
                              language_name, looks_contaminated, normalize_for_compare,
                              parse_model_ids, sanitize_translation, split_batches,
                              strip_numbering, system_prompt)
from engine.ocr import OcrEngine, merge_vertical_columns
from engine.compose import cover_boxes

APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)
CFG = {"llm": {"provider": "mock", "mock": {}}}


class Transport(QWebChannelAbstractTransport):
    def __init__(self):
        super().__init__()
        self.messages = []

    @Slot(dict)
    def sendMessage(self, message):
        self.messages.append(dict(message))

    def receive(self, message):
        self.messageReceived.emit(message, self)


def connect_channel(bridge):
    channel = QWebChannel()
    channel.setPropertyUpdateInterval(0)
    channel.registerObject("bridge", bridge)
    transport = Transport()
    channel.connectTo(transport)
    transport.receive({"type": 3, "id": 1})
    info = next(m["data"]["bridge"] for m in transport.messages if m.get("type") == 10)
    signals = {name: index for name, index in info["signals"]}
    for index in set(signals.values()):
        transport.receive({"type": 7, "object": "bridge", "signal": index})
    return channel, transport, signals


def wait_for(predicate, limit=4):
    deadline = time.monotonic() + limit
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    if not predicate():
        raise AssertionError("等待后台任务超时")


def png_bytes(color="white"):
    output = io.BytesIO()
    Image.new("RGB", (100, 60), color).save(output, "PNG")
    return output.getvalue()


class FakePipeline:
    def __init__(self, fail=False):
        self.fail = fail
        self.cfg = copy.deepcopy(CFG)
        self.translator = SimpleNamespace(cfg=self.cfg)
        self.last_image = None

    def process(self, image, progress=None):
        self.last_image = image.copy()
        progress("测试：OCR 完成")
        if self.fail:
            raise RuntimeError("模拟翻译服务失败")
        boxes = [{"x": 1, "y": 1, "w": 20, "h": 20, "text": "こんにちは", "score": 1.0}]
        return Image.new("RGB", (100, 60), "white"), boxes, {"こんにちは": "你好"}


class BridgeTests(unittest.TestCase):
    def run_job(self, fail=False):
        b = Bridge(copy.deepcopy(CFG))
        b._pipeline = FakePipeline(fail)
        b._selected_region = [0, 0, 100, 60]
        b._selected_png = png_bytes("red")
        channel, transport, signals = connect_channel(b)
        count = b.metaObject().methodCount()
        b.translate_page("0,0,100,60")
        wait_for(lambda: not b._busy)
        APP.processEvents()
        self.assertEqual(count, b.metaObject().methodCount(), "运行中不能增删桥的 Qt 方法")
        delivered = [m for m in transport.messages if m.get("type") == 1]
        results = [json.loads(m["args"][0]) for m in delivered if m.get("signal") == signals["result_ready"]]
        logs = [m["args"][0] for m in delivered if m.get("signal") == signals["log_line"]]
        self.assertEqual(len(results), 1, "结果必须真正进入网页传输层，不能只测 Python 信号")
        self.assertTrue(any("开始翻译" in s for s in logs))
        b.shutdown()
        return results[0], logs, b._pipeline.last_image

    def test_success_reaches_webchannel(self):
        result, logs, image = self.run_job()
        self.assertTrue(result["ok"])
        self.assertEqual(result["translations"]["こんにちは"], "你好")
        self.assertTrue(any("完成：共 1" in s for s in logs))
        self.assertTrue(result["image"].startswith("data:image/png;base64,"))
        self.assertEqual(tuple(image[0, 0]), (0, 0, 255), "应使用冻结截图而不是实时屏幕")

    def test_failure_reaches_webchannel(self):
        result, logs, _ = self.run_job(True)
        self.assertFalse(result["ok"])
        self.assertIn("模拟翻译服务失败", result["error"])
        self.assertTrue(any("出错" in s for s in logs))

    def test_invalid_region(self):
        b = Bridge(copy.deepcopy(CFG))
        channel, transport, signals = connect_channel(b)
        b.translate_page("1,2,0,4")
        APP.processEvents()
        self.assertTrue(any(m.get("signal") == signals["result_ready"] for m in transport.messages if m.get("type") == 1))

    def test_no_duplicate_jobs_or_provider_mutation(self):
        b = Bridge(copy.deepcopy(CFG))
        b._busy = True
        b.set_provider("paid")
        b.translate_page("0,0,100,60")
        self.assertIsNone(b._worker)
        self.assertEqual(b._cfg["llm"]["provider"], "mock")

    def test_source_language_and_grain_respect_job_guards(self):
        b = Bridge(copy.deepcopy(CFG))
        b.set_source_language("ja")
        b.set_grain(False)
        self.assertEqual(b._cfg["llm"]["source_lang"], "ja")
        self.assertFalse(b._cfg["inpaint"]["grain"])
        b.set_source_language("invalid")
        self.assertEqual(b._cfg["llm"]["source_lang"], "ja")
        for flag in ("_busy", "_reading_active"):
            setattr(b, flag, True)
            b.set_source_language("en")
            b.set_grain(True)
            self.assertEqual(b._cfg["llm"]["source_lang"], "ja")
            self.assertFalse(b._cfg["inpaint"]["grain"])
            setattr(b, flag, False)

    def test_set_prompts_stores_and_can_reset(self):
        b = Bridge(copy.deepcopy(CFG))
        logs = []
        b.log_line.connect(logs.append)
        b.set_prompts("把$from翻成$to", "")
        self.assertEqual(b._cfg["llm"]["prompts"]["system"], "把$from翻成$to")
        self.assertIsNone(b._cfg["llm"]["prompts"]["batch"])
        self.assertTrue(any("系统" in line for line in logs))
        # 空串表示恢复内置默认
        b.set_prompts("", "")
        self.assertIsNone(b._cfg["llm"]["prompts"]["system"])
        self.assertTrue(any("内置默认" in line for line in logs))
        b._busy = True
        b.set_prompts("不许改", "")
        self.assertIsNone(b._cfg["llm"]["prompts"]["system"], "任务进行中不许改提示词")

    def test_set_inpaint_switches_method_and_respects_busy(self):
        b = Bridge(copy.deepcopy(CFG))
        logs = []
        b.log_line.connect(logs.append)
        b.set_inpaint("lama")
        self.assertEqual(b._cfg["inpaint"]["method"], "lama")
        self.assertTrue(any("抹字方式已切换为 lama" in line for line in logs))
        b.set_inpaint("不存在的算法")
        self.assertEqual(b._cfg["inpaint"]["method"], "lama", "非法值不能写进配置")
        b._busy = True
        b.set_inpaint("cv2")
        self.assertEqual(b._cfg["inpaint"]["method"], "lama", "任务进行中不许改")

    def test_capture_cancel_restores_controls(self):
        b = Bridge(copy.deepcopy(CFG))
        channel, transport, signals = connect_channel(b)
        b._picking = True
        b._on_region_cancelled()
        self.assertFalse(b._picking)
        sent = [m for m in transport.messages if m.get("type") == 1]
        self.assertTrue(any(m.get("signal") == signals["busy_changed"] and m["args"] == [False] for m in sent))

    @staticmethod
    def paid_bridge():
        cfg = copy.deepcopy(CFG)
        cfg["llm"]["provider"] = "paid"
        cfg["llm"]["paid"] = {"base_url": "https://api.deepseek.com/v1",
                              "api_key": "sk-test", "model": "deepseek-chat"}
        return Bridge(cfg)

    def test_test_connection_discovers_models(self):
        b = self.paid_bridge()
        logs, payloads = [], []
        b.log_line.connect(logs.append)
        b.models_ready.connect(payloads.append)

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]}

        with patch("engine.translate.requests.get", return_value=Response()) as fake_get:
            b.test_connection()
            wait_for(lambda: b._models_worker is None)
            APP.processEvents()
        fake_get.assert_called_once_with("https://api.deepseek.com/v1/models",
                                         headers={"Authorization": "Bearer sk-test"}, timeout=(5, 15))
        data = json.loads(payloads[-1])
        self.assertTrue(data["ok"])
        self.assertEqual(data["models"], ["deepseek-chat", "deepseek-reasoner"])
        self.assertEqual(data["current"], "deepseek-chat")
        self.assertTrue(any("连接成功" in line for line in logs))

    def test_test_connection_failure_and_guard(self):
        import requests
        b = self.paid_bridge()
        logs, payloads = [], []
        b.log_line.connect(logs.append)
        b.models_ready.connect(payloads.append)
        with patch("engine.translate.requests.get",
                   side_effect=requests.exceptions.ConnectionError("boom")):
            b.test_connection()
            wait_for(lambda: b._models_worker is None)
            APP.processEvents()
        data = json.loads(payloads[-1])
        self.assertFalse(data["ok"])
        self.assertIn("连不上", data["error"])
        self.assertTrue(any("连接测试失败" in line for line in logs))
        # 没有 base_url 的通道（如 mock）给出明确提示，而不是抛未处理异常
        empty = Bridge(copy.deepcopy(CFG))
        payloads2 = []
        empty.models_ready.connect(payloads2.append)
        empty.test_connection()
        wait_for(lambda: empty._models_worker is None)
        APP.processEvents()
        data2 = json.loads(payloads2[-1])
        self.assertFalse(data2["ok"])
        self.assertIn("base_url", data2["error"])
        # 忙碌时拒绝发起，也不留下线程
        b._busy = True
        b.test_connection()
        self.assertIsNone(b._models_worker)

    def test_set_model_writes_channel_and_respects_busy(self):
        b = Bridge(copy.deepcopy(CFG))
        logs = []
        b.log_line.connect(logs.append)
        b.set_model("qwen2.5:7b")
        self.assertEqual(b._cfg["llm"]["mock"]["model"], "qwen2.5:7b")
        self.assertTrue(any("模型已切换为 qwen2.5:7b" in line for line in logs))
        b.set_model("  ")
        self.assertEqual(b._cfg["llm"]["mock"]["model"], "qwen2.5:7b", "空模型名不能写进配置")
        b._busy = True
        b.set_model("其他模型")
        self.assertEqual(b._cfg["llm"]["mock"]["model"], "qwen2.5:7b", "任务进行中不许切换")

    def test_apply_preset_fills_channel_and_replaces_options(self):
        b = Bridge(copy.deepcopy(CFG))
        b._cfg["llm"]["mock"]["model"] = "old"
        b._cfg["llm"]["options"] = {"temperature": 0.9}
        logs = []
        b.log_line.connect(logs.append)
        preset = {"name": "DeepSeek（推荐）", "provider": "paid",
                  "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat",
                  "options": {"temperature": 0, "thinking": {"type": "disabled"}}}
        b.apply_preset(preset)
        self.assertEqual(b._cfg["llm"]["provider"], "paid")
        self.assertEqual(b._cfg["llm"]["paid"]["base_url"], "https://api.deepseek.com/v1")
        self.assertEqual(b._cfg["llm"]["paid"]["model"], "deepseek-chat")
        self.assertEqual(b._cfg["llm"]["options"],
                         {"temperature": 0, "thinking": {"type": "disabled"}}, "options 整体替换而不是叠加")
        self.assertTrue(any("已套用预设「DeepSeek（推荐）」" in line for line in logs))
        # 预设字典被后续修改不能串进配置
        preset["options"]["temperature"] = 5
        self.assertEqual(b._cfg["llm"]["options"]["temperature"], 0)
        # 非法 provider 忽略；忙碌时拒绝
        b.apply_preset({"name": "坏预设", "provider": "mock", "model": "x"})
        self.assertEqual(b._cfg["llm"]["provider"], "paid")
        b._busy = True
        b.apply_preset({"name": "本地", "provider": "free", "base_url": "http://localhost:11434/v1",
                        "model": "qwen2.5:7b"})
        self.assertEqual(b._cfg["llm"]["provider"], "paid")
        self.assertFalse(any("已套用预设「本地」" in line for line in logs))


class FakeScreen:
    def __init__(self, x=0, y=0):
        self.geometry_rect = QRect(x, y, 1600, 1000)

    def geometry(self):
        return self.geometry_rect


def event(x, y, button=Qt.LeftButton):
    return SimpleNamespace(position=lambda: SimpleNamespace(toPoint=lambda: QPoint(x, y)), button=lambda: button)


class SelectorTests(unittest.TestCase):
    def selector(self, x=0, y=0):
        image = QImage(2400, 1500, QImage.Format_RGB32)
        image.fill(QColor("white"))
        painter = QPainter(image)
        painter.fillRect(QRect(800, 400, 600, 600), QColor("red"))
        painter.end()
        image.setDevicePixelRatio(1.5)
        return RegionSelector(FakeScreen(x, y), image)

    def test_dpi_and_release_endpoint(self):
        w = self.selector()
        got = []
        w.region_selected.connect(got.append)
        w.mousePressEvent(event(400, 300))
        # 不发送 move，验证松开终点同样生效。
        w.mouseReleaseEvent(event(1300, 850))
        self.assertEqual(got, [[600, 450, 1350, 825]])
        self.assertEqual((w.selected_image().width(), w.selected_image().height()), (1350, 825))

    def test_reverse_drag_and_screen_origin(self):
        w = self.selector(-2400, 100)
        w.mousePressEvent(event(1300, 850))
        w._current = QPoint(400, 300)
        self.assertEqual(w._region(), [-1800, 550, 1350, 825])
        w.close()

    def test_clamp_to_screen(self):
        w = self.selector()
        w.mousePressEvent(event(1500, 900))
        w._current = QPoint(1800, 1200)
        self.assertEqual(w._region(), [2250, 1350, 150, 150])
        w.close()

    def test_close_right_click_and_escape_cancel_once(self):
        for action in (lambda w: w.close(), lambda w: w.mousePressEvent(event(0, 0, Qt.RightButton)),
                       lambda w: w.keyPressEvent(SimpleNamespace(key=lambda: Qt.Key_Escape))):
            w = self.selector()
            got = []
            w.cancelled.connect(lambda: got.append(True))
            action(w)
            w.close()
            self.assertEqual(got, [True])

    def test_selected_content_is_not_resampled_or_zoomed(self):
        w = self.selector()
        w._origin, w._current = QPoint(400, 250), QPoint(1100, 800)
        w._selecting = True
        canvas = QImage(w.size(), QImage.Format_RGB32)
        canvas.fill(QColor("black"))
        w.render(canvas)
        self.assertEqual(canvas.pixelColor(650, 500).name(), "#ff0000")
        self.assertEqual(canvas.pixelColor(1000, 500).name(), "#ffffff")
        self.assertNotEqual(canvas.pixelColor(1400, 500).name(), "#ffffff", "选框外必须压暗")
        w.close()


class OcrLayoutTests(unittest.TestCase):
    def column(self, text, x, y=80, w=44, h=335):
        return {"x": x, "y": y, "w": w, "h": h, "text": text, "score": 0.99, "vertical": True}

    def test_vertical_sentence_merges_right_to_left(self):
        left = self.column("しい一日", 502, y=149, w=46, h=200)
        right = self.column("今日はとても楽", 552)
        merged = merge_vertical_columns([left, right])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "今日はとても楽しい一日")
        self.assertEqual((merged[0]["x"], merged[0]["y"], merged[0]["w"], merged[0]["h"]), (502, 80, 94, 335))
        self.assertEqual(left["text"], "しい一日", "不得修改原始框")

    def test_merged_columns_preserve_background_between_columns(self):
        boxes = [self.column("右列", 150, y=10, w=40, h=120), self.column("左列", 100, y=30, w=40, h=80)]
        image = np.full((160, 240, 3), 255, dtype=np.uint8)
        image[15:20, 110:120] = (0, 0, 255)
        out = cover_boxes(image, merge_vertical_columns(boxes), {"inpaint": {"pad": 0}})
        self.assertEqual(tuple(out[16, 115]), (0, 0, 255), "合并框内但原文框外的背景不得被抹掉")

    def test_width_mismatch_and_gap_threshold(self):
        self.assertEqual(len(merge_vertical_columns([self.column("窄列", 100, w=20), self.column("宽列", 130, w=44)])), 2)
        self.assertEqual(len(merge_vertical_columns([self.column("左列", 100), self.column("右列", 171)])), 2)
        self.assertEqual(len(merge_vertical_columns([self.column("左列", 100), self.column("右列", 170)])), 1)

    def test_overlapping_columns_still_merge(self):
        """真实检测框会横向轻微重叠（实测 gap=-4px），必须照样合并成一句。"""
        left = self.column("しい一日", 500, y=150, w=52, h=195)
        right = self.column("今日はとても楽", 548, y=74, w=54, h=344)
        merged = merge_vertical_columns([left, right])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "今日はとても楽しい一日")

    def test_excessive_horizontal_overlap_is_not_merged(self):
        """横向重叠超过半个字宽更像同一列被检出两次，不能合并。"""
        boxes = [self.column("上の台詞", 100, w=50), self.column("下の台詞", 105, w=50)]
        self.assertEqual(merge_vertical_columns(boxes), boxes)

    def test_merge_can_be_disabled(self):
        left = self.column("しい一日", 502, y=149, w=46, h=200)
        right = self.column("今日はとても楽", 552)
        polys = [[[b["x"], b["y"]], [b["x"] + b["w"], b["y"]],
                  [b["x"] + b["w"], b["y"] + b["h"]], [b["x"], b["y"] + b["h"]]] for b in (left, right)]
        fake = SimpleNamespace(predict=lambda image: [{"rec_texts": [left["text"], right["text"]],
                                                       "rec_scores": [0.99, 0.99], "rec_polys": polys}])
        ocr = OcrEngine.__new__(OcrEngine)
        ocr._ocr, ocr._merge_columns = fake, False
        self.assertEqual(len(ocr.detect(None)), 2)
        ocr._merge_columns = True
        self.assertEqual(ocr.detect(None)[0]["text"], "今日はとても楽しい一日")

    def test_separate_bubbles_are_not_merged(self):
        boxes = [self.column("おやすみなさい", 138), self.column("今日はとても楽", 552)]
        self.assertEqual(merge_vertical_columns(boxes), boxes)

    def test_nonoverlapping_columns_are_not_merged(self):
        boxes = [self.column("上の台詞", 502, y=20, h=120), self.column("下の台詞", 552, y=300, h=120)]
        self.assertEqual(merge_vertical_columns(boxes), boxes)

    def test_horizontal_boxes_are_not_merged(self):
        boxes = [self.column("横排对白", 502, h=40), self.column("另一句话", 552, h=40)]
        for box in boxes:
            box["vertical"] = False
        self.assertEqual(merge_vertical_columns(boxes), boxes)

    def test_three_columns_and_single_character(self):
        boxes = [self.column("左端", 100), self.column("右端", 200), self.column("中间", 150)]
        self.assertEqual(merge_vertical_columns(boxes)[0]["text"], "右端中间左端")
        self.assertEqual(len(merge_vertical_columns([self.column("あ", 100), self.column("言葉", 150)])), 2)
        self.assertEqual(merge_vertical_columns([]), [])

    def test_staggered_columns_are_not_chained(self):
        """错位的多列不得经中间列级联合成一大团：逐列偏离超过 0.6 字宽即断开。"""
        boxes = [self.column("三の列", 100, y=100, h=200),
                 self.column("二の列", 152, y=50, h=200),
                 self.column("一の列", 204, y=0, h=200)]
        merged = merge_vertical_columns(boxes)
        self.assertEqual([b["text"] for b in merged], ["一の列", "二の列", "三の列"])

    def test_slight_stagger_still_merges(self):
        """同一句各列检测框顶边有小幅参差（不超过 0.6 字宽）时仍要合成一句。"""
        boxes = [self.column("三の列", 100, y=10, h=200),
                 self.column("二の列", 152, y=20, h=200),
                 self.column("一の列", 204, y=0, h=200)]
        merged = merge_vertical_columns(boxes)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "一の列二の列三の列")
        self.assertEqual((merged[0]["x"], merged[0]["y"], merged[0]["w"], merged[0]["h"]),
                         (100, 0, 148, 220))

    def test_merged_output_is_sorted_by_position(self):
        """输出统一按 (上, 左) 排序：与输入顺序无关，呈现顺序稳定。"""
        boxes = [self.column("下の右", 552, y=400, h=120),
                 self.column("下の左", 500, y=400, h=120),
                 self.column("上の右", 552, y=10, h=120),
                 self.column("上の左", 500, y=10, h=120)]
        merged = merge_vertical_columns(boxes)
        self.assertEqual([b["text"] for b in merged], ["上の右上の左", "下の右下の左"])


class PromptTemplateTests(unittest.TestCase):
    """提示词模板化：占位符插值、config 覆盖、默认回退。"""

    def config(self, **prompts):
        return {"llm": {"provider": "free", "prompts": prompts,
                        "free": {"base_url": "http://x/v1", "api_key": "k", "model": "m"}}}

    def test_placeholders_are_substituted(self):
        from engine.translate import system_prompt
        cfg = self.config(system="把$from翻成$to（识别为$detect）")
        self.assertEqual(system_prompt("ja", cfg), "把日文翻成简体中文（识别为日文）")

    def test_blank_prompt_falls_back_to_default(self):
        from engine.translate import DEFAULT_SYSTEM_PROMPT, system_prompt
        for value in (None, "", "   "):
            with self.subTest(value=value):
                cfg = self.config(system=value)
                rendered = system_prompt("ja", cfg)
                self.assertIn("$from", DEFAULT_SYSTEM_PROMPT, "默认模板本身应带占位符")
                self.assertIn("$to", DEFAULT_SYSTEM_PROMPT, "默认模板本身应带占位符")
                self.assertNotIn("$", rendered, "渲染后不应残留任何占位符")
                self.assertIn("日文", rendered, "回退默认后 $from 应被替换成语言名")
                self.assertIn("简体中文", rendered, "回退默认后 $to 应被替换成目标语言")

    def test_missing_prompts_section_uses_default(self):
        from engine.translate import system_prompt
        self.assertIn("日文", system_prompt("ja", {"llm": {"provider": "free"}}))

    def test_whitespace_only_template_falls_back_to_default(self):
        """全是空格的模板必须当没填，否则用户手滑敲空格就会拿到空提示词。"""
        from engine.translate import batch_prompt, system_prompt
        cfg = self.config(system="  \t ", batch="\n  \n")
        self.assertIn("日文", system_prompt("ja", cfg))
        self.assertIn("1. Hello", batch_prompt("ja", "1. Hello", cfg))
        # 全空格的系统提示绝不能进请求体
        self.assertNotEqual(system_prompt("ja", cfg).strip(), "")

    def test_batch_prompt_carries_numbered_text(self):
        from engine.translate import batch_prompt
        cfg = self.config(batch="翻译$from：\n$text\n只给译文")
        rendered = batch_prompt("en", "1. Hello\n2. World", cfg)
        self.assertIn("1. Hello", rendered)
        self.assertIn("英文", rendered)
        self.assertTrue(rendered.startswith("翻译"), rendered)
        self.assertTrue(rendered.endswith("只给译文"), rendered)

    def test_braces_in_source_text_do_not_break_render(self):
        """用 replace 而非 format，原文里的花括号不能炸掉渲染。"""
        from engine.translate import batch_prompt
        cfg = self.config(batch="$text")
        rendered = batch_prompt("ja", "1. 变量{a}和{b}", cfg)
        self.assertIn("{a}", rendered)

    def test_custom_prompt_reaches_the_request_payload(self):
        """自定义提示词必须真的进请求体，不能只存在配置里。"""
        cfg = self.config(system="自定义系统提示$from", batch="自定义批量：$text")
        t = Translator(cfg)
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "你好"}}]}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return _Resp()

        with patch("engine.translate.requests.post", side_effect=fake_post):
            t.translate(["こんにちは"])
        messages = captured["payload"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("自定义系统提示", messages[0]["content"])
        self.assertIn("日文", messages[0]["content"], "$from 应在系统提示里被替换")
        self.assertIn("自定义批量：", messages[1]["content"])
        self.assertIn("こんにちは", messages[1]["content"], "$text 应带进用户提示")

    def test_custom_prompt_changes_cache_signature(self):
        """改过提示词就该换缓存桶，否则新旧模板的译文会混用。"""
        plain = Translator(self.config())
        custom = Translator(self.config(system="自定义系统提示$from"))
        changed = Translator(self.config(system="另一个系统提示$from"))
        batch_changed = Translator(self.config(batch="另一套批量：$text"))
        self.assertNotEqual(plain._cache_signature(), custom._cache_signature())
        self.assertNotEqual(custom._cache_signature(), changed._cache_signature())
        self.assertNotEqual(custom._cache_signature(), batch_changed._cache_signature())
        # 没改提示词的实例签名必须稳定、可复现
        self.assertEqual(plain._cache_signature(), Translator(self.config())._cache_signature())
        # 全空格的"没填"不参与签名，与未配置一致
        self.assertEqual(plain._cache_signature(),
                         Translator(self.config(system="  \t "))._cache_signature())

    def test_builtin_default_change_changes_cache_signature(self):
        """改内置默认提示词也必须换缓存桶。

        缓存签名只算"用户覆盖"是不够的：内置默认提示词一改，老缓存会一直压着
        新提示词不生效，用户看到的还是旧译文。
        """
        from engine import translate as tr
        plain = Translator(self.config())
        before = plain._cache_signature()
        original = tr.DEFAULT_SYSTEM_PROMPT
        try:
            tr.DEFAULT_SYSTEM_PROMPT = original + "（新增约束）"
            after = Translator(self.config())._cache_signature()
        finally:
            tr.DEFAULT_SYSTEM_PROMPT = original
        self.assertNotEqual(before, after, "内置默认提示词变了，签名必须跟着变")
        self.assertEqual(Translator(self.config())._cache_signature(), before,
                         "复原内置默认后签名应回到原值")

    def test_prompt_change_bypasses_old_cache(self):
        """落盘级验证：同一句在改提示词后必须重新调用模型，不命中旧缓存。"""
        with tempfile.TemporaryDirectory() as folder:
            cache_section = {"enabled": True, "path": os.path.join(folder, "c.json")}
            first_cfg = self.config(system="提示词甲$from")
            first_cfg["cache"] = cache_section
            first = Translator(first_cfg)
            with patch.object(first, "_chat", return_value="你好") as chat:
                self.assertEqual(first.translate(["こんにちは"]), {"こんにちは": "你好"})
                self.assertEqual(chat.call_count, 1)

            # 同一提示词的新实例：命中缓存，零调用
            same_cfg = self.config(system="提示词甲$from")
            same_cfg["cache"] = cache_section
            same = Translator(same_cfg)
            with patch.object(same, "_chat", return_value="不该被调用") as chat:
                self.assertEqual(same.translate(["こんにちは"]), {"こんにちは": "你好"})
                self.assertEqual(chat.call_count, 0)

            # 改了提示词：旧缓存作废，必须重新调用模型
            changed_cfg = self.config(system="提示词乙$from")
            changed_cfg["cache"] = cache_section
            changed = Translator(changed_cfg)
            with patch.object(changed, "_chat", return_value="你好啊") as chat:
                self.assertEqual(changed.translate(["こんにちは"]), {"こんにちは": "你好啊"})
                self.assertEqual(chat.call_count, 1)


class DefaultPromptGuardTests(unittest.TestCase):
    """内置默认提示词的内容约束：修好的坑不能再被改回去。

    实测踩过两个坑：模型把假名转写成罗马字（こそい -> kosoi）、把"括号注释"
    当补充说明塞进译文；以及拿到 OCR 垃圾碎片时自己编一句对白。
    """

    def test_default_prompt_forbids_romanization(self):
        from engine.translate import DEFAULT_SYSTEM_PROMPT
        self.assertIn("罗马字", DEFAULT_SYSTEM_PROMPT,
                      "必须明确禁止用罗马字转写假名")
        self.assertIn("kosoi", DEFAULT_SYSTEM_PROMPT, "最好留一个具体反例给模型看")

    def test_default_prompt_forbids_asides(self):
        from engine.translate import DEFAULT_SYSTEM_PROMPT
        self.assertIn("括号", DEFAULT_SYSTEM_PROMPT, "必须禁止加括号注释或补充说明")

    def test_default_prompt_tells_model_not_to_invent_text(self):
        from engine.translate import DEFAULT_SYSTEM_PROMPT
        self.assertIn("无法辨认", DEFAULT_SYSTEM_PROMPT,
                      "必须说明无法辨认的碎片不要编造成对白")
        self.assertIn("保留原文", DEFAULT_SYSTEM_PROMPT)

    def test_default_prompt_keeps_output_contract_and_placeholders(self):
        from engine.translate import DEFAULT_SYSTEM_PROMPT
        self.assertIn("只输出译文", DEFAULT_SYSTEM_PROMPT, "输出契约不能丢")
        self.assertIn("$from", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("$to", DEFAULT_SYSTEM_PROMPT)

    def test_default_prompt_still_covers_line_mapping(self):
        from engine.translate import DEFAULT_SYSTEM_PROMPT
        self.assertIn("对应关系", DEFAULT_SYSTEM_PROMPT,
                      "批量翻译时各条对应关系必须保持，不能合并或调换")


class TranslatorTests(unittest.TestCase):
    def translator(self, batch_size=None):
        llm = {"provider": "free",
               "free": {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "test"}}
        if batch_size is not None:
            llm["batch_size"] = batch_size
        return Translator({"llm": llm})

    def test_batch_and_deduplicate(self):
        t = self.translator()
        with patch.object(t, "_chat", return_value="1. 你好\n2. 再见") as chat:
            self.assertEqual(t.translate(["こんにちは", "さようなら", "こんにちは"]), {"こんにちは": "你好", "さようなら": "再见"})
            self.assertEqual(chat.call_count, 1)

    def test_fallback_reports_progress(self):
        t = self.translator()
        logs = []
        with patch.object(t, "_chat", side_effect=["格式错", "你好", "再见"]):
            result = t.translate(["こんにちは", "さようなら"], progress=logs.append)
        self.assertEqual(len(result), 2)
        self.assertTrue(any("第 2/2" in s for s in logs))

    def test_batch_runtime_failure_falls_back_to_per_sentence(self):
        """批量超时/格式错误时逐句重发，不让整页翻译失败。"""
        t = self.translator()
        logs = []
        with patch.object(t, "_chat",
                          side_effect=[RuntimeError("翻译请求超时"), "你好", "再见"]) as chat:
            result = t.translate(["こんにちは", "さようなら"], progress=logs.append)
        self.assertEqual(chat.call_count, 3, "一次批量失败 + 两句逐句重试")
        self.assertEqual(result, {"こんにちは": "你好", "さようなら": "再见"})
        self.assertTrue(any("改为逐句翻译" in s for s in logs))

    def test_batch_http_5xx_falls_back_to_per_sentence(self):
        import requests
        t = self.translator()
        response = requests.Response()
        response.status_code = 500
        error = requests.exceptions.HTTPError(response=response)
        logs = []
        with patch.object(t, "_chat", side_effect=[error, "你好", "再见"]) as chat:
            result = t.translate(["こんにちは", "さようなら"], progress=logs.append)
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result["こんにちは"], "你好")
        self.assertTrue(any("HTTP 500" in s for s in logs))

    def test_401_and_connection_errors_skip_the_fallback(self):
        import requests
        t = self.translator()
        unauthorized = requests.Response()
        unauthorized.status_code = 401
        error = requests.exceptions.HTTPError(response=unauthorized)
        with patch.object(t, "_chat", side_effect=error) as chat:
            with self.assertRaisesRegex(RuntimeError, "鉴权失败"):
                t.translate(["こんにちは"])
        self.assertEqual(chat.call_count, 1, "401 应立刻报错，不做逐句重试")

        t2 = self.translator()
        with patch.object(t2, "_chat", side_effect=requests.exceptions.ConnectionError()) as chat2:
            with self.assertRaisesRegex(RuntimeError, "连不上"):
                t2.translate(["こんにちは"])
        self.assertEqual(chat2.call_count, 1, "连不上服务也不该逐句重试")

    def test_split_batches_isolates_long_and_noise(self):
        short = ["こんにちは", "さようなら"]
        long_text = "これはとても長い台詞で、一度に読むと息が続かないほどの長さの文章です"
        noise = "！？"
        batches = split_batches([*short, long_text, noise, "おはよう"], size=2)
        self.assertEqual(batches, [[short[0], short[1]], [long_text], [noise], ["おはよう"]])

    def test_split_batches_caps_batch_size_and_keeps_order(self):
        texts = [f"句子{i}" for i in range(20)]
        batches = split_batches(texts, size=8)
        self.assertEqual([len(b) for b in batches], [8, 8, 4])
        self.assertEqual([t for batch in batches for t in batch], texts)

    def test_default_batch_size_stays_small(self):
        """默认批量不能大：7B 模型批量一大就串扰（假名转写成罗马字、混进英文）。"""
        from engine.translate import _BATCH_SIZE
        self.assertLessEqual(_BATCH_SIZE, 4,
                             "实测每批 8 句时开始出现 こそり→kosoi、没Thing 这类污染")
        self.assertGreaterEqual(_BATCH_SIZE, 1)

    def test_batch_size_config_overrides_default(self):
        for value, expected in ((1, 1), (4, 4), (0, 1), (999, 32), ("3", 3)):
            with self.subTest(value=value):
                self.assertEqual(self.translator(value).batch_size, expected)

    def test_bad_batch_size_falls_back_to_default(self):
        from engine.translate import _BATCH_SIZE
        for bad in (None, "abc", float("nan")):
            with self.subTest(bad=bad):
                self.assertEqual(self.translator(bad).batch_size, _BATCH_SIZE)

    def test_timeout_report(self):
        import requests
        t = self.translator()
        with patch("engine.translate.requests.post", side_effect=requests.exceptions.ReadTimeout):
            with self.assertRaisesRegex(RuntimeError, "超时"):
                t.translate(["こんにちは"])

    def test_parse_region(self):
        self.assertEqual(parse_region("[-10，20，300，400]"), [-10, 20, 300, 400])
        for text in ("1,2,0,3", "1,2,-1,3", "a,b,c,d", "1,2,3"):
            self.assertIsNone(parse_region(text))

    def test_strip_numbering(self):
        self.assertEqual(strip_numbering("1. 你好"), "你好")
        self.assertEqual(strip_numbering("10. 序，然..."), "序，然...")
        self.assertEqual(strip_numbering("19、_tmp_show_test.py"), "_tmp_show_test.py")
        self.assertEqual(strip_numbering("（3）早安"), "早安")
        self.assertEqual(strip_numbering("3.5 倍速"), "3.5 倍速")  # 小数不能被当成编号
        self.assertEqual(strip_numbering("没有编号"), "没有编号")
        self.assertEqual(strip_numbering("12."), "12.")

    def test_strip_numbering_with_index_handles_digit_initial_text(self):
        """真实日志：模型给数字开头的译文加了序号，启发式保护把它全放过了。"""
        cases = [(11, "11. 8N", "8N"), (17, "17. 5.7.6", "5.7.6"),
                 (20, "20. 4.重点验三", "4.重点验三"), (21, "21. 8", "8"),
                 (29, "29. 6.点「样式", "6.点「样式"), (47, "47. 1个后台", "1个后台")]
        for index, line, expected in cases:
            with self.subTest(line=line):
                self.assertEqual(strip_numbering(line, index), expected)
        # "①" 的 isdigit() 也是 True，同样要按序号剥掉
        self.assertEqual(strip_numbering("27. ①", 27), "①")

    def test_strip_numbering_with_index_keeps_non_matching_lines(self):
        """序号对不上说明这行本来就以数字开头，不能误剥。"""
        self.assertEqual(strip_numbering("45. 2", 44), "45. 2")

    def test_digit_initial_translations_survive_batch(self):
        """批量翻译里译文本身以数字开头时，序号要剥掉、内容要保住。"""
        t = self.translator(8)
        texts = ["こんにちは", "おはよう", "ありがとう"]
        reply = "1. 8月\n2. 5.7.6\n3. 1个后台"
        with patch.object(t, "_chat", return_value=reply):
            result = t.translate(texts)
        self.assertEqual(result, {"こんにちは": "8月", "おはよう": "5.7.6", "ありがとう": "1个后台"})

    def test_multidigit_numbering_stripped_in_batch(self):
        t = self.translator(8)
        texts = [f"こんにちは{i}" for i in range(1, 12)]
        # 微批次把 11 句拆成 8+3 两批，模型每批从 1 重新编号，同样要剥得干净。
        replies = ["\n".join(f"{i}. 译{i}" for i in range(1, 9)),
                   "\n".join(f"{i}. 译{i + 8}" for i in range(1, 4))]
        with patch.object(t, "_chat", side_effect=replies) as chat:
            result = t.translate(texts)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["こんにちは10"], "译10")
        self.assertEqual(result["こんにちは11"], "译11")
        self.assertEqual(result["こんにちは1"], "译1")

    def test_detect_language(self):
        self.assertEqual(detect_language(["おはよう"]), "ja")
        self.assertEqual(detect_language(["カタカナ"]), "ja")
        self.assertEqual(detect_language(["私は漫画が好き"]), "ja")   # 汉字+假名仍是日文
        self.assertEqual(detect_language(["안녕하세요"]), "ko")
        self.assertEqual(detect_language(["你好世界"]), "zh")
        self.assertEqual(detect_language(["收尾", "5个文件已更改"]), "zh")
        self.assertEqual(detect_language(["Hello there"]), "en")
        self.assertEqual(detect_language(["Привет"]), "ru")
        self.assertEqual(detect_language(["7"]), "unknown")
        self.assertEqual(language_name("en"), "英文")

    def test_chinese_region_skips_llm(self):
        t = self.translator()
        logs = []
        with patch.object(t, "_chat") as chat:
            self.assertEqual(t.translate(["收尾", "5个文件已更改"], progress=logs.append), {})
            chat.assert_not_called()
        self.assertTrue(any("已经是中文，无需翻译" in s for s in logs))

    def test_english_region_is_translated(self):
        """英文也要翻成中文，不能被“非日文”挡掉。"""
        t = self.translator()
        logs = []
        with patch.object(t, "_chat", return_value="早上好") as chat:
            result = t.translate(["Good morning"], progress=logs.append)
        self.assertEqual(result, {"Good morning": "早上好"})
        self.assertTrue(chat.called)
        self.assertEqual(chat.call_args.args[4], "en", "应把识别出的语言传给模型")
        self.assertTrue(any("英文" in s for s in logs))

    def test_source_lang_can_be_fixed(self):
        """config 固定 source_lang 时不再自动判断。"""
        cfg = {"llm": {"provider": "free", "source_lang": "ja",
                       "free": {"base_url": "http://x/v1", "api_key": "k", "model": "m"}}}
        t = Translator(cfg)
        self.assertEqual(t.source_language(["你好"]), "ja")

    def test_system_prompt_follows_language(self):
        from engine.translate import system_prompt
        self.assertIn("日文", system_prompt("ja"))
        self.assertIn("英文", system_prompt("en"))

    def test_all_echo_warns(self):
        t = self.translator()
        logs = []
        with patch.object(t, "_chat", return_value="こんにちは"):
            t.translate(["こんにちは"], progress=logs.append)
        self.assertTrue(any("原样返回" in s for s in logs))

    def test_punctuation_only_echo_counted_as_unchanged(self):
        """只改标点的"翻译"也算回声，不能当成有效译文。"""
        t = self.translator()
        logs = []
        with patch.object(t, "_chat", return_value="こんにちは。"):
            result = t.translate(["こんにちは！"], progress=logs.append)
        self.assertEqual(result, {"こんにちは！": "こんにちは。"})
        self.assertTrue(any("原样返回" in s for s in logs))


class ContaminationGuardTests(unittest.TestCase):
    """批量串扰守卫：译文里冒出原文没有的拉丁字母时，整句重发一次。

    实测 qwen2.5:7b 批量翻译时会串扰——把假名转写成罗马字（こそり→kosoi）、
    混进英文单词（大丈夫大丈夫ぅ→"没事没Thing"）。这些字母原文里没有，一抓一个准。
    """

    def translator(self, batch_size=2):
        return Translator({"llm": {"provider": "free", "batch_size": batch_size,
                                   "source_lang": "ja",
                                   "free": {"base_url": "http://x/v1",
                                            "api_key": "k", "model": "m"}}})

    def test_helper_detects_romanization_and_english(self):
        self.assertTrue(looks_contaminated("こそり行こう", "去kosoi吧"))
        self.assertTrue(looks_contaminated("大丈夫大丈夫ぅ！", "没事没Thing！"))

    def test_helper_ignores_clean_and_latin_sources(self):
        self.assertFalse(looks_contaminated("こんにちは", "你好"))
        self.assertFalse(looks_contaminated("こんにちは", ""), "空译文不算污染")
        # 原文本身有字母时不算污染：该保留的就该保留（"OK" 翻成 "OK" 是对的）
        self.assertFalse(looks_contaminated("OK、行こう", "OK，走吧"))
        self.assertFalse(looks_contaminated("BGM かけて", "放BGM"))

    def test_has_latin_letters_covers_accented(self):
        self.assertTrue(has_latin_letters("café"))
        self.assertFalse(has_latin_letters("你好，世界"))
        self.assertFalse(has_latin_letters("こんにちは"))

    def test_contaminated_line_is_retranslated_alone(self):
        t = self.translator()
        replies = ["1. 那之后去kosoi吧？\n2. 没问题！", "那下次再一起去怎么样？"]
        with patch.object(t, "_chat", side_effect=replies) as chat:
            result = t.translate(["じゃああとでこそり行ちぉか?", "大丈夫大丈夫ぅ！"])
        self.assertEqual(chat.call_count, 2, "一批 + 一句重发")
        self.assertEqual(result["じゃああとでこそり行ちぉか?"], "那下次再一起去怎么样？")
        self.assertEqual(result["大丈夫大丈夫ぅ！"], "没问题！")

    def test_clean_batch_triggers_no_retry(self):
        t = self.translator()
        with patch.object(t, "_chat", return_value="1. 你好\n2. 再见") as chat:
            result = t.translate(["こんにちは", "さようなら"])
        self.assertEqual(chat.call_count, 1, "干净译文不该多花一次调用")
        self.assertEqual(result, {"こんにちは": "你好", "さようなら": "再见"})

    def test_retry_that_stays_contaminated_keeps_original(self):
        t = self.translator()
        replies = ["1. 去kosoi吧\n2. 好的", "还是kosoi"]
        with patch.object(t, "_chat", side_effect=replies) as chat:
            result = t.translate(["こそり行こう", "はい"])
        self.assertEqual(chat.call_count, 2, "只重发一次，不做第三轮")
        self.assertEqual(result["こそり行こう"], "去kosoi吧", "重发仍带字母就保留原结果")

    def test_guard_skipped_when_source_has_latin(self):
        t = self.translator()
        with patch.object(t, "_chat", return_value="1. OK，走吧\n2. 好的") as chat:
            t.translate(["OK、行こう", "はい"])
        self.assertEqual(chat.call_count, 1, "原文本身有字母，不该误判成污染")

    def test_guard_repairs_more_than_one_line(self):
        t = self.translator()
        replies = ["1. 没事没Thing！\n2. 去kosoi吧", "没问题！", "那下次再一起去怎么样？"]
        with patch.object(t, "_chat", side_effect=replies) as chat:
            result = t.translate(["大丈夫大丈夫ぅ！", "じゃああとでこそり行ちぉか?"])
        self.assertEqual(chat.call_count, 3, "一批 + 两句各重发一次")
        self.assertEqual(result["大丈夫大丈夫ぅ！"], "没问题！")
        self.assertEqual(result["じゃああとでこそり行ちぉか?"], "那下次再一起去怎么样？")


class SanitizeTranslationTests(unittest.TestCase):
    """模型输出清洗：思考块/JSON 包装/装饰剥离，无效内容返回空串。"""

    def test_think_block_and_analysis_tags_stripped(self):
        # 尖括号用 \u003c/\u003e 转义写，避免被当成标记吞掉
        self.assertEqual(
            sanitize_translation("\u003cthinking\u003e让我想想\u003c/thinking\u003e你好"), "你好")
        self.assertEqual(sanitize_translation("<analysis>分析中</analysis>早上好"), "早上好")

    def test_unclosed_think_discards_the_rest(self):
        """没闭合的思考块说明其后仍在思考，整段作废而不是当译文用。"""
        self.assertEqual(sanitize_translation("\u003cthink\u003e还在分析"), "")

    def test_think_leak_markers_discard_whole_line(self):
        self.assertEqual(sanitize_translation("Thinking Process: 1. 分析原文"), "")
        self.assertEqual(sanitize_translation("Analyze the Request:\n原文是日文"), "")

    def test_json_wrapping_extracted(self):
        self.assertEqual(sanitize_translation('{"translation": "你好"}'), "你好")
        self.assertEqual(sanitize_translation('{"translatedText": "再见"}'), "再见")
        self.assertEqual(sanitize_translation('{"text": "早上好"}'), "早上好")
        self.assertEqual(sanitize_translation('{"output": "晚安"}'), "晚安")
        self.assertEqual(
            sanitize_translation('{"data": {"translations": [{"translatedText": "谢谢"}]}}'), "谢谢")

    def test_malformed_and_fenced_json_fallback(self):
        self.assertEqual(sanitize_translation('{"translation": "你好'), "你好")          # 被截断
        self.assertEqual(sanitize_translation('```json\n{"translation": "你好"}\n```'), "你好")
        self.assertEqual(sanitize_translation('{"translation": ""}'), "",
                         "空字段的 JSON 壳不能当译文渲染")

    def test_decorations_stripped(self):
        self.assertEqual(sanitize_translation('**"你好"**'), "你好")
        self.assertEqual(sanitize_translation("『你好』"), "『你好』", "日文引号是内容不是装饰")

    def test_invalid_results(self):
        for bad in ("", "   ", "...", "…", None, "  ...  "):
            self.assertEqual(sanitize_translation(bad), "")

    def test_batch_blank_line_keeps_alignment(self):
        """某行被清洗成空时必须留空占位，否则后续译文全部错位。"""
        t = TranslatorTests().translator(8)
        reply = "1. 你好\n2. Thinking Process: 分析\n3. 再见"
        with patch.object(t, "_chat", return_value=reply) as chat:
            result = t.translate(["こんにちは", "さようなら", "ありがとう"])
        self.assertEqual(chat.call_count, 1, "行数仍是 3，不应触发逐句兜底")
        self.assertEqual(result, {"こんにちは": "你好", "さようなら": "", "ありがとう": "再见"})


class GarbledTranslationTests(unittest.TestCase):
    def test_replacement_char_and_question_storm(self):
        self.assertTrue(is_garbled_translation("你好\ufffd"))
        self.assertTrue(is_garbled_translation("？？？？"))
        self.assertFalse(is_garbled_translation("真的吗？"))

    def test_repeated_characters(self):
        self.assertTrue(is_garbled_translation("好好好好好好好"))
        self.assertFalse(is_garbled_translation("你好好久不见"))

    def test_high_frequency_characters(self):
        self.assertTrue(is_garbled_translation("你好你好你好你好你好你好"))   # 单字频率超 60%
        self.assertTrue(is_garbled_translation("搬运搬运搬运搬运搬运搬运搬运搬运搬运搬运"))

    def test_punctuation_and_empty_are_garbled(self):
        self.assertTrue(is_garbled_translation("..."))
        self.assertTrue(is_garbled_translation("！？"))
        self.assertTrue(is_garbled_translation(""))

    def test_normal_translation_passes(self):
        self.assertFalse(is_garbled_translation("早上好，今天天气不错"))
        self.assertFalse(is_garbled_translation("嗯……"))

    def test_normalizer_ignores_punctuation(self):
        self.assertEqual(normalize_for_compare("おはよう。！"), normalize_for_compare("おはよう"))


class TranslationCacheTests(unittest.TestCase):
    def test_punctuation_only_echo_is_not_cached(self):
        """只改标点的回声不入缓存，避免把失败结果固化成"翻译"。"""
        from engine.cache import TranslationCache

        with tempfile.TemporaryDirectory() as folder:
            cfg = {"cache": {"enabled": True, "path": os.path.join(folder, "c.json")}}
            cache = TranslationCache(cfg)
            changed = cache.remember("sig", {"おはよう。": "おはよう", "ありがとう": "谢谢"})
            self.assertEqual(changed, 1)
            self.assertEqual(cache.lookup("sig", ["おはよう。", "ありがとう"]), {"ありがとう": "谢谢"})


class RejectLogTests(unittest.TestCase):
    """失败样本留存：清洗丢弃 / 回声 / 乱码按天存 JSONL，写入失败不抛错。"""

    def free_cfg(self, folder):
        cfg = {"llm": {"provider": "free",
                       "free": {"base_url": "http://x/v1", "api_key": "k", "model": "m"}}}
        cfg["rejects"] = {"enabled": True, "dir": folder}
        return cfg

    def read_entries(self, folder):
        entries = []
        for name in os.listdir(folder):
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                entries.extend(json.loads(line) for line in handle if line.strip())
        return entries

    def test_missing_section_is_disabled(self):
        from engine.rejects import RejectLog
        self.assertFalse(RejectLog({}).enabled)
        self.assertFalse(RejectLog({"llm": {}}).enabled)

    def test_disabled_switch_writes_nothing(self):
        from engine.rejects import RejectLog
        with tempfile.TemporaryDirectory() as folder:
            RejectLog({"rejects": {"enabled": False, "dir": folder}}).record("清洗后为空", "原文", "返回值")
            self.assertEqual(os.listdir(folder), [])

    def test_entries_are_appended_by_day(self):
        from engine.rejects import RejectLog
        with tempfile.TemporaryDirectory() as folder:
            log = RejectLog({"rejects": {"enabled": True, "dir": folder}})
            log.record("清洗后为空", "おはよう", "Thinking Process: ...")
            log.record("乱码或退化输出", "こんにちは", "你好你好你好你好你好你好")
            names = os.listdir(folder)
            self.assertEqual(len(names), 1)
            self.assertRegex(names[0], r"^rejects-\d{8}\.jsonl$")
            entries = self.read_entries(folder)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["reason"], "清洗后为空")
            self.assertEqual(entries[0]["source"], "おはよう")
            self.assertEqual(entries[0]["model"], "Thinking Process: ...")
            self.assertTrue(entries[0]["time"])

    def test_write_failure_is_silent(self):
        """目录被文件占位时 write 必须降级为静默——留存不能拖垮翻译。"""
        from engine.rejects import RejectLog
        with tempfile.TemporaryDirectory() as folder:
            blocker = os.path.join(folder, "blocker")
            with open(blocker, "w", encoding="utf-8") as handle:
                handle.write("x")
            RejectLog({"rejects": {"enabled": True, "dir": blocker}}).record("清洗后为空", "原文", "返回值")

    def test_sanitized_output_is_recorded(self):
        """思考泄漏被清洗后留样：能回看模型到底吐了什么。"""
        with tempfile.TemporaryDirectory() as folder:
            t = Translator(self.free_cfg(folder))
            with patch.object(t, "_chat", return_value="Thinking Process: 我先分析一下"):
                result = t.translate(["こんにちは"])
            self.assertEqual(result, {"こんにちは": ""})
            entries = self.read_entries(folder)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["reason"], "清洗后为空")
            self.assertEqual(entries[0]["source"], "こんにちは")
            self.assertIn("Thinking Process", entries[0]["model"])

    def test_echo_is_recorded(self):
        """原样返回的回声也留样：据此判断是提示词问题还是文本本身的问题。"""
        with tempfile.TemporaryDirectory() as folder:
            t = Translator(self.free_cfg(folder))
            with patch.object(t, "_chat", return_value="こんにちは"):
                result = t.translate(["こんにちは"])
            self.assertEqual(result, {"こんにちは": "こんにちは"})
            entries = self.read_entries(folder)
            self.assertEqual([e["reason"] for e in entries], ["原样返回"])
            self.assertEqual(entries[0]["source"], "こんにちは")
            self.assertEqual(entries[0]["model"], "こんにちは")


class GlossaryTests(unittest.TestCase):
    """术语表：config.yaml 的 glossary 段注入系统提示词，并参与缓存签名。"""

    def cfg(self, glossary=None):
        cfg = {"llm": {"provider": "free",
                       "free": {"base_url": "http://x/v1", "api_key": "k", "model": "m"}}}
        if glossary is not None:
            cfg["glossary"] = glossary
        return cfg

    def test_system_prompt_carries_glossary(self):
        prompt = system_prompt("ja", self.cfg({"主人公": "主角", "ライバル": "劲敌"}))
        self.assertIn("主人公→主角", prompt)
        self.assertIn("ライバル→劲敌", prompt)
        self.assertNotIn("术语表", system_prompt("ja", self.cfg({})))
        self.assertNotIn("术语表", system_prompt("ja", None))

    def test_glossary_reaches_the_request_payload(self):
        t = Translator(self.cfg({"先生": "老师"}))
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "你好"}}]}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return _Resp()

        with patch("engine.translate.requests.post", side_effect=fake_post):
            t.translate(["こんにちは"])
        self.assertIn("先生→老师", captured["payload"]["messages"][0]["content"])

    def test_glossary_changes_cache_signature(self):
        plain = Translator(self.cfg())
        with_glossary = Translator(self.cfg({"主人公": "主角"}))
        changed = Translator(self.cfg({"主人公": "男主"}))
        self.assertNotEqual(plain._cache_signature(), with_glossary._cache_signature())
        self.assertNotEqual(with_glossary._cache_signature(), changed._cache_signature())
        self.assertEqual(plain._cache_signature(), Translator(self.cfg())._cache_signature())


class ModelDiscoveryTests(unittest.TestCase):
    """模型发现纯函数：base_url 推导 /models、解析模型 id 列表。"""

    def test_build_models_url_variants(self):
        self.assertEqual(build_models_url("https://api.deepseek.com/v1"),
                         "https://api.deepseek.com/v1/models")
        self.assertEqual(build_models_url("https://api.deepseek.com/v1/"),
                         "https://api.deepseek.com/v1/models")
        self.assertEqual(build_models_url("https://api.deepseek.com/v1/chat/completions"),
                         "https://api.deepseek.com/v1/models")
        self.assertEqual(build_models_url("https://api.deepseek.com/chat/completions"),
                         "https://api.deepseek.com/models")
        self.assertEqual(build_models_url("http://localhost:11434/v1/models"),
                         "http://localhost:11434/v1/models")
        for value in ("", "   ", None):
            self.assertEqual(build_models_url(value), "")

    def test_parse_model_ids_variants(self):
        payload = {"data": [{"id": "b"}, {"id": "a"}, {"id": "a"}, {"name": "c"},
                            {"model": "d"}, "e", 42, {"id": "  "}]}
        self.assertEqual(parse_model_ids(payload), ["a", "b", "c", "d", "e"])
        self.assertEqual(parse_model_ids([{"id": "x"}, {"id": "x"}]), ["x"])
        for value in (None, {"data": "nope"}, {"data": None}, 7):
            self.assertEqual(parse_model_ids(value), [])

    def test_fetch_models_requires_base_url(self):
        with self.assertRaisesRegex(RuntimeError, "base_url"):
            fetch_models({"base_url": ""})
        with self.assertRaisesRegex(RuntimeError, "base_url"):
            fetch_models({})

    def test_fetch_models_sends_auth_and_parses(self):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"id": "m1"}, {"id": "m2"}]}

        with patch("engine.translate.requests.get", return_value=Response()) as fake_get:
            url, models = fetch_models({"base_url": "https://x/v1", "api_key": "k"})
        fake_get.assert_called_once_with("https://x/v1/models",
                                         headers={"Authorization": "Bearer k"}, timeout=(5, 15))
        self.assertEqual(url, "https://x/v1/models")
        self.assertEqual(models, ["m1", "m2"])
        # 本地服务没有 key 时不带 Authorization 头
        with patch("engine.translate.requests.get", return_value=Response()) as fake_get:
            fetch_models({"base_url": "http://localhost:11434/v1"})
        fake_get.assert_called_once_with("http://localhost:11434/v1/models",
                                         headers={}, timeout=(5, 15))


class RecognizeOnlyTests(unittest.TestCase):
    """只识别不翻译：诊断模式不能抹字，也不能把原文交给翻译服务。"""

    def build(self, texts):
        from engine.pipeline import Pipeline

        class FakeOcr:
            def detect(self, image):
                return [{"x": 5, "y": 5, "w": 30, "h": 10, "text": t, "score": 0.87, "vertical": False}
                        for t in texts]

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.cfg = copy.deepcopy(CFG)
        pipeline.ocr = FakeOcr()
        pipeline.translator = Translator(copy.deepcopy(CFG))
        return pipeline

    def test_recognize_never_covers_or_translates(self):
        pipeline = self.build(["こんにちは", "おはよう"])
        image = np.full((40, 60, 3), 200, dtype=np.uint8)
        image[5:15, 5:35] = 30           # 原文是深色像素
        with patch("engine.pipeline.cover_boxes") as cover, \
                patch.object(pipeline.translator, "translate") as translate:
            result, boxes, language = pipeline.recognize_only(image)
        cover.assert_not_called()
        translate.assert_not_called()
        self.assertEqual(len(boxes), 2)
        self.assertEqual(language, "ja")
        # 标注图只在框上画线，原文像素必须原样保留
        self.assertEqual(tuple(np.array(result)[7, 7]), (30, 30, 30))

    def test_recognize_reports_language_and_boxes_to_progress(self):
        pipeline = self.build(["東堂容太"])
        logs = []
        with patch("engine.pipeline.cover_boxes") as cover:
            _, boxes, language = pipeline.recognize_only(
                np.full((40, 60, 3), 200, np.uint8), progress=logs.append)
        cover.assert_not_called()
        self.assertEqual([b["text"] for b in boxes], ["東堂容太"])
        self.assertIn(language, ("ja", "zh"))
        self.assertTrue(any("识别到 1 个文字块" in s for s in logs))

    def test_recognize_only_returns_annotation_image(self):
        from engine.pipeline import recognize_image
        cfg = copy.deepcopy(CFG)
        image = np.full((40, 60, 3), 200, np.uint8)
        with patch("engine.pipeline.Pipeline.__init__", return_value=None), \
                patch("engine.pipeline.Pipeline.recognize", return_value=(
                    [{"x": 3, "y": 3, "w": 20, "h": 10, "text": "こんにちは", "score": 0.9, "vertical": False}], 0.1)):
            result, boxes, language = recognize_image(image, cfg)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(language, "ja")
        annotated = np.array(result)[:, :, ::-1]
        # 标注框画上去了，但原文区域底色仍是原来的 200
        self.assertFalse(np.array_equal(annotated, image), "应画出标注框")
        self.assertEqual(tuple(annotated[20, 50]), (200, 200, 200), "非目标区域不能被改色")


class PipelineTests(unittest.TestCase):
    def build(self, texts):
        from engine.pipeline import Pipeline

        class FakeOcr:
            def detect(self, image):
                return [{"x": 5, "y": 5, "w": 30, "h": 10, "text": t, "score": 0.99, "vertical": False}
                        for t in texts]

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.cfg = copy.deepcopy(CFG)
        pipeline.ocr = FakeOcr()
        pipeline.translator = Translator(copy.deepcopy(CFG))
        return pipeline

    def test_chinese_region_is_left_untouched(self):
        """框到中文时不能抹字：原图返回、无文字块，悬浮层才不会盖上白块。"""
        pipeline = self.build(["收尾", "5个文件已更改"])
        image = np.full((40, 60, 3), 200, dtype=np.uint8)
        logs = []
        result, boxes, translations = pipeline.process(image, progress=logs.append)
        self.assertEqual(boxes, [])
        self.assertEqual(translations, {})
        self.assertTrue(np.array_equal(np.array(result), image[:, :, ::-1]), "原图不应被修改")
        self.assertTrue(any("已经是中文，无需翻译" in s for s in logs))

    def test_english_region_is_not_skipped(self):
        """英文选区要正常走抹字+翻译，不能像中文那样被跳过。"""
        pipeline = self.build(["Good morning"])
        with patch.object(pipeline.translator, "translate", return_value={"Good morning": "早上好"}) as translate:
            image = np.full((40, 60, 3), 200, dtype=np.uint8)
            logs = []
            result, boxes, translations = pipeline.process(image, progress=logs.append)
        self.assertTrue(translate.called)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(translations, {"Good morning": "早上好"})
        self.assertTrue(any("英文" in s for s in logs))

    def test_mixed_region_drops_chinese_and_symbol_blocks(self):
        """真实日志：漫画选区里混进界面文字，中文/纯数字块被塞给外文模型后
        模型回一句"请提供英文对白"并盖掉原文。这些块必须保持原样。"""
        pipeline = self.build(["たちまち彼は、天使の視界のかぎり見る。", "文件夹", "2026/8/5 13:23"])
        with patch.object(pipeline.translator, "translate",
                          return_value={"たちまち彼は、天使の視界のかぎり見る。": "不久，他看到了天使所能见到的一切。"}) as translate:
            image = np.full((40, 60, 3), 200, dtype=np.uint8)
            logs = []
            result, boxes, translations = pipeline.process(image, progress=logs.append)
        self.assertEqual([b["text"] for b in boxes], ["たちまち彼は、天使の視界のかぎり見る。"])
        self.assertEqual(translate.call_args.args[0], ["たちまち彼は、天使の視界のかぎり見る。"],
                         "中文与纯数字块不应进入翻译")
        self.assertTrue(any("跳过 2 个" in s for s in logs))

    def test_japanese_page_drops_symbol_soup_blocks(self):
        """日文页里既无假名也无汉字的碎片是 OCR 噪声，不能送去翻译。

        实测拟声词「グッ」被读成 "$\\r$"（conf 0.63，越过 min_score），模型把它
        译成 "1." 盖上去——真实内容被抹掉换成垃圾，比不翻更糟。
        """
        pipeline = self.build(["グッ", "$\\r$", "こんにちは"])
        with patch.object(pipeline.translator, "translate",
                          return_value={"グッ": "咕", "こんにちは": "你好"}) as translate:
            logs = []
            _, boxes, _ = pipeline.process(np.full((40, 60, 3), 200, np.uint8), progress=logs.append)
        self.assertEqual(translate.call_args.args[0], ["グッ", "こんにちは"],
                         "无日文内容的碎片不应进入翻译")
        self.assertEqual([b["text"] for b in boxes], ["グッ", "こんにちは"])
        self.assertTrue(any("跳过 1 个" in s for s in logs))

    def test_english_page_keeps_latin_blocks(self):
        """英文页的纯拉丁块是正常内容，不能被"无日文内容"过滤误伤。"""
        pipeline = self.build(["Good morning", "OK"])
        with patch.object(pipeline.translator, "translate",
                          return_value={"Good morning": "早上好", "OK": "好"}) as translate:
            _, boxes, _ = pipeline.process(np.full((40, 60, 3), 200, np.uint8))
        self.assertEqual(sorted(translate.call_args.args[0]), ["Good morning", "OK"])
        self.assertEqual(len(boxes), 2)

    def test_fixed_japanese_keeps_han_only_blocks(self):
        pipeline = self.build(["東堂容太", "少々", "123"])
        pipeline.cfg["llm"]["source_lang"] = "ja"
        pipeline.translator.cfg = pipeline.cfg
        with patch.object(pipeline.translator, "translate", return_value={"東堂容太": "东堂容太", "少々": "稍微"}) as translate:
            _, boxes, _ = pipeline.process(np.full((40, 60, 3), 200, np.uint8))
        self.assertEqual(translate.call_args.args[0], ["東堂容太", "少々"])
        self.assertEqual([b["text"] for b in boxes], ["東堂容太", "少々"])

    def test_fixed_language_applies_before_mixed_page_filter(self):
        pipeline = self.build(["少々", "こんにちは"])
        pipeline.cfg["llm"]["source_lang"] = "ja"
        with patch.object(pipeline.translator, "translate", return_value={"少々": "稍微", "こんにちは": "你好"}) as translate:
            _, boxes, _ = pipeline.process(np.full((40, 60, 3), 200, np.uint8))
        self.assertEqual(len(boxes), 2)
        self.assertIn("少々", translate.call_args.args[0])

    def test_missing_translation_keeps_original_pixels(self):
        pipeline = self.build(["こんにちは"])
        original = np.full((40, 60, 3), 200, np.uint8)
        original[5:15, 5:35] = 0
        with patch.object(pipeline.translator, "translate", return_value={}), patch("engine.pipeline.cover_boxes", wraps=cover_boxes) as cover:
            result, boxes, _ = pipeline.process(original)
        self.assertEqual(boxes, [])
        self.assertEqual(cover.call_args.args[1], [])
        self.assertTrue(np.array_equal(np.array(result), original[:, :, ::-1]))

    def test_garbled_translation_is_dropped_and_logged(self):
        """乱码译文不抹字，同时留样到 rejects JSONL，方便回看模型吐了什么。"""
        from engine.rejects import RejectLog
        with tempfile.TemporaryDirectory() as folder:
            pipeline = self.build(["こんにちは"])
            pipeline.cfg["rejects"] = {"enabled": True, "dir": folder}
            pipeline.translator.rejects = RejectLog(pipeline.cfg)
            garbled = "你好你好你好你好你好你好"
            with patch.object(pipeline.translator, "translate", return_value={"こんにちは": garbled}):
                _, boxes, _ = pipeline.process(np.full((40, 60, 3), 200, np.uint8))
            self.assertEqual(boxes, [], "乱码译文不能回填")
            names = os.listdir(folder)
            self.assertEqual(len(names), 1)
            with open(os.path.join(folder, names[0]), encoding="utf-8") as handle:
                entry = json.loads(handle.readline())
            self.assertEqual(entry["reason"], "乱码或退化输出")
            self.assertEqual(entry["source"], "こんにちは")
            self.assertEqual(entry["model"], garbled)

    def test_region_of_only_symbols_leaves_image_untouched(self):
        """整块都是数字/符号时同样不抹字。"""
        pipeline = self.build(["2026/8/5 13:23", "5.7.6"])
        image = np.full((40, 60, 3), 200, dtype=np.uint8)
        result, boxes, translations = pipeline.process(image)
        self.assertEqual(boxes, [])
        self.assertEqual(translations, {})
        self.assertTrue(np.array_equal(np.array(result), image[:, :, ::-1]))


class OcrOptionsTests(unittest.TestCase):
    def test_mobile_models_are_passed(self):
        options = build_ocr_options({"ocr": {"det_model": "PP-OCRv5_mobile_det",
                                             "rec_model": "PP-OCRv5_mobile_rec"}})
        self.assertEqual(options["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(options["text_recognition_model_name"], "PP-OCRv5_mobile_rec")

    def test_null_models_fall_back_to_lang(self):
        options = build_ocr_options({"ocr": {"lang": "japan", "det_model": None, "rec_model": None}})
        self.assertNotIn("text_detection_model_name", options)
        self.assertNotIn("text_recognition_model_name", options)
        self.assertEqual(options["lang"], "japan")

    def test_mkldnn_stays_disabled(self):
        """oneDNN 在本机 paddlepaddle 3.3.1 上一推理就抛 NotImplementedError。"""
        self.assertFalse(build_ocr_options({})["enable_mkldnn"])

    def test_tunables_are_plumbed(self):
        options = build_ocr_options({"ocr": {"textline_orientation": False, "det_limit_side_len": 640}})
        self.assertFalse(options["use_textline_orientation"])
        self.assertEqual(options["text_det_limit_side_len"], 640)

    def test_project_config_uses_mobile_models(self):
        """防止 config.yaml 被改回 medium 档把速度打回 13 秒。"""
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        options = build_ocr_options(cfg)
        self.assertEqual(options.get("text_detection_model_name"), "PP-OCRv5_mobile_det")
        self.assertEqual(options.get("text_recognition_model_name"), "PP-OCRv5_mobile_rec")


class CoverGrainTests(unittest.TestCase):
    """深色/带网点背景上的抹字质量：纯 inpaint 会留下一块没有纹理的平滑补丁。"""

    def page(self):
        """造一张带颗粒噪声与横向排线的页面，黑字直接印在画面上（非纯色气泡）。"""
        rng = np.random.default_rng(7)
        h, w = 200, 460
        base = np.full((h, w), 165, np.uint8)
        for i in range(0, h, 3):
            base[i:i + 1, :] = np.clip(base[i:i + 1, :].astype(int) - rng.integers(20, 55, (1, w)), 0, 255)
        base = cv2.GaussianBlur(base, (3, 3), 0)
        gray = np.clip(base.astype(float) + rng.normal(0, 26, (h, w)), 0, 255).astype(np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        image[40:70, 60:400] = 5          # 模拟一行黑字（足够暗，必然触发 inpaint 回退）
        return image

    BOX = {"x": 58, "y": 38, "w": 344, "h": 34, "text": "あ", "score": 0.95, "vertical": False}
    CFG = {"inpaint": {"method": "cover", "pad": 3, "cover_ring_width": 5,
                       "cover_flat_std": 12, "grain": True}}

    def covered(self, **overrides):
        from engine.compose import cover_boxes
        cfg = copy.deepcopy(self.CFG)
        cfg["inpaint"].update(overrides)
        return cover_boxes(self.page(), [copy.deepcopy(self.BOX)], cfg)

    @staticmethod
    def texture_ratio(image, box):
        """补丁区梯度能量 / 周围背景梯度能量：接近 1 才说明纹理没被抹平。"""
        patch = image[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]
        ref = image[box["y"] - 30:box["y"] - 6, box["x"]:box["x"] + box["w"]]
        inner = cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        outer = cv2.Laplacian(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        return inner / outer if outer else 0.0

    def test_inpaint_alone_flattens_texture(self):
        """记录旧行为：没有颗粒时补丁纹理几乎为 0，就是一块死平的色块。"""
        self.assertLess(self.texture_ratio(self.covered(grain=False), self.BOX), 0.1)

    def test_optional_grain_is_bounded_not_texture_reconstruction(self):
        """实验颗粒只准轻微增纹理，不能再用接近背景梯度能量作为质量目标。"""
        flat = self.covered(grain=False).astype(np.int16)
        grained = self.covered(grain=True).astype(np.int16)
        delta = (grained - flat)[45:65, 70:390]
        self.assertGreater(float(delta.std()), 0.5)
        self.assertLess(float(delta.std()), 9.0)

    def test_default_cover_never_calls_random_grain(self):
        cfg = copy.deepcopy(self.CFG)
        del cfg["inpaint"]["grain"]
        with patch("engine.compose._add_grain", side_effect=AssertionError("默认不许制造噪点")):
            image = cover_boxes(self.page(), [copy.deepcopy(self.BOX)], cfg)
        self.assertEqual(image.shape, self.page().shape)

    def test_text_and_border_edges_do_not_become_grain(self):
        from engine.compose import _crop_with_mask, _grain_std
        image = np.full((100, 160, 3), 245, np.uint8)
        image[35:65, 60:100] = 0  # 模拟贴边的黑字
        image[:, 112:114] = 0    # 相邻气泡边线
        crop, mask, _, _ = _crop_with_mask(image, 60, 35, 100, 65, 18)
        self.assertLess(_grain_std(crop, mask), 0.5)

    def test_grain_strength_has_a_hard_limit(self):
        from engine.compose import _crop_with_mask, _grain_std
        image = self.page()
        crop, mask, _, _ = _crop_with_mask(image, 60, 40, 400, 70, 15)
        self.assertLessEqual(_grain_std(crop, mask), 8.0)

    def test_color_alignment_remains_enabled_without_grain(self):
        from engine.compose import _inpaint_region
        image = np.full((100, 160, 3), 160, np.uint8)
        fake_fixed = np.full_like(image[25:75, 45:115], 195)
        with patch("engine.compose.cv2.inpaint", return_value=fake_fixed):
            result = _inpaint_region(image, 50, 30, 110, 70, 5, grain=False)
        self.assertEqual(int(np.median(result)), 160)

    def test_project_config_disables_artificial_grain(self):
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        self.assertFalse(cfg["inpaint"]["grain"])

    def test_grain_does_not_leave_original_text_behind(self):
        """补颗粒不能把原文又露出来：补丁区的暗像素比例应与周围背景相当。

        原图有字时该比例是 87%，抹干净后补丁与背景都在 2% 上下。
        """
        image = self.covered(grain=True)
        box = self.BOX
        patch = cv2.cvtColor(image[box["y"]:box["y"] + box["h"],
                                   box["x"]:box["x"] + box["w"]], cv2.COLOR_BGR2GRAY)
        ref = cv2.cvtColor(image[0:box["y"] - 6, box["x"]:box["x"] + box["w"]], cv2.COLOR_BGR2GRAY)
        original = cv2.cvtColor(self.page()[box["y"]:box["y"] + box["h"],
                                         box["x"]:box["x"] + box["w"]], cv2.COLOR_BGR2GRAY)
        self.assertGreater(float((original < 96).mean()), 0.5, "构造的样图应确实压着黑字")
        self.assertLess(float((patch < 96).mean()), 0.01, "不能保留原字，也不能人为制造大量黑点")

    def test_script_fallback_still_used_when_lingua_absent(self):
        """降级路径（无 lingua）必须保持旧行为，不能因为装了库才可用。"""
        from engine import translate
        with patch.object(translate, "_lingua_detector", return_value=None):
            self.assertEqual(translate.detect_language(["おはよう"]), "ja")
            self.assertEqual(translate.detect_language(["안녕하세요"]), "ko")
            self.assertEqual(translate.detect_language(["你好世界"]), "zh")
            self.assertEqual(translate.detect_language(["Hello"]), "en")
            self.assertEqual(translate.detect_language(["7"]), "unknown")

    def test_lingua_is_consulted_for_han_only_text(self):
        """纯汉字必须问统计模型：这是区间计数唯一答不了的情况。"""
        from engine import translate

        class Value:
            def __init__(self, name, value):
                self.language = SimpleNamespace(name=name)
                self.value = value

        fake = SimpleNamespace(compute_language_confidence_values=lambda text: [
            Value("JAPANESE", 0.91), Value("CHINESE", 0.08)])
        with patch.object(translate, "_lingua_detector", return_value=fake):
            self.assertEqual(translate.detect_language(["東堂容太"]), "ja")

    def test_lingua_cannot_rescue_short_han_names(self):
        """诚实记录能力边界：短汉字人名中日通用，统计模型也判不出来。

        lingua 实测对 '東堂容太' 给出 日文=0.000，只能判成中文。所以阅读日漫
        真正可靠的手段是手动固定 llm.source_lang=ja，不能依赖自动识别。
        """
        from engine import translate

        class Value:
            def __init__(self, name, value):
                self.language = SimpleNamespace(name=name)
                self.value = value

        fake = SimpleNamespace(compute_language_confidence_values=lambda text: [
            Value("CHINESE", 1.0), Value("JAPANESE", 0.0)])
        with patch.object(translate, "_lingua_detector", return_value=fake):
            self.assertEqual(translate.detect_language(["東堂容太"]), "zh")
        # 而 pipeline 在固定日文时会绕过这层过滤，保证人名仍被翻译
        with patch.object(translate, "_lingua_detector", return_value=None):
            self.assertEqual(translate.detect_language(["東堂容太"]), "zh")

    def test_latin_never_consults_lingua(self):
        """拉丁文本必然是英文，不必为它启动统计模型。"""
        from engine import translate
        with patch.object(translate, "_lingua_detector") as detector:
            self.assertEqual(translate.detect_language(["Thank you"]), "en")
            detector.assert_not_called()

    def test_low_confidence_falls_back_to_script(self):
        from engine import translate

        class Value:
            def __init__(self, name, value):
                self.language = SimpleNamespace(name=name)
                self.value = value

        fake = SimpleNamespace(compute_language_confidence_values=lambda text: [
            Value("JAPANESE", 0.40), Value("CHINESE", 0.35)])
        with patch.object(translate, "_lingua_detector", return_value=fake):
            self.assertEqual(translate.detect_language(["東堂容太"]), "zh", "置信度不够就退回计数法")

    def test_kana_short_circuits_before_lingua(self):
        """假名是日文独有信号，不该再花代价调统计模型。"""
        from engine import translate
        with patch.object(translate, "_lingua_detector") as detector:
            self.assertEqual(translate.detect_language(["こんにちは"]), "ja")
            detector.assert_not_called()

    def test_grain_is_deterministic(self):
        """颗粒用位置派生的固定种子，同输入必须同输出（否则实时模式每帧都在抖）。"""
        self.assertTrue(np.array_equal(self.covered(), self.covered()))

    def test_grain_stays_achromatic(self):
        """颗粒只加在亮度上。逐通道独立加噪会在灰阶漫画上留下彩色麻点，比不加还显眼。"""
        image = self.covered(grain=True)
        box = self.BOX
        patch = image[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]].astype(np.int16)
        chroma = float(np.abs(patch - patch.mean(axis=2, keepdims=True)).mean())
        self.assertLess(chroma, 1.0, "灰阶底上的补丁不该出现彩色像素")

    def test_flat_cover_also_gets_grain(self):
        """纯色气泡那条分支同样要补颗粒，否则白底上也会盖出一块死平的方块。"""
        rng = np.random.default_rng(3)
        flat = np.clip(220 + rng.normal(0, 7, (200, 460, 3)), 0, 255).astype(np.uint8)
        flat[80:120, 60:400] = 10          # 黑字压在纯色底上
        from engine.compose import cover_boxes
        cfg = copy.deepcopy(self.CFG)
        box = {"x": 58, "y": 78, "w": 344, "h": 44, "text": "あ", "score": 0.95, "vertical": False}
        out = cover_boxes(flat, [box], cfg)
        patch = out[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]
        ref = out[box["y"] - 30:box["y"] - 6, box["x"]:box["x"] + box["w"]]
        inner = cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        outer = cv2.Laplacian(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        self.assertGreater(inner, 0, "手动开启的细颗粒应有效")
        self.assertLess(float(patch.astype(np.float32).std()), 8.0, "纯色气泡不应被补成脏麻点")


class RenderStyleTests(unittest.TestCase):
    CFG = {"render": {"font_path": "C:/Windows/Fonts/msyh.ttc", "min_font_size": 10, "line_gap": 1.2}}
    BOX = {"x": 10, "y": 10, "w": 200, "h": 90, "text": "こんにちは", "score": 1.0, "vertical": False}

    def render(self, extra, background="white"):
        from engine.render import render_texts
        cfg = copy.deepcopy(self.CFG)
        cfg["render"].update(extra)
        image = Image.new("RGB", (240, 120), background)
        return render_texts(image, [copy.deepcopy(self.BOX)], {"こんにちは": "你好世界"}, cfg)

    @staticmethod
    def ink(image):
        return sum(1 for pixel in image.getdata() if pixel != (255, 255, 255))

    def test_font_scale_enlarges_text(self):
        small, big = self.render({"font_scale": 1.0}), self.render({"font_scale": 1.8})
        self.assertGreater(self.ink(big), self.ink(small))

    def test_color_is_applied(self):
        image = self.render({"color": "#ff0000", "auto_color": False})
        self.assertTrue(any(r > 150 and g < 100 and b < 100 for r, g, b in image.getdata()))

    def test_default_color_stays_black(self):
        image = self.render({})
        self.assertTrue(any(r < 80 and g < 80 and b < 80 for r, g, b in image.getdata()))

    def test_invalid_style_falls_back_to_default(self):
        from engine.render import text_style
        self.assertEqual(text_style({"render": {"font_scale": "abc", "color": None}}),
                         {"scale": 1.0, "color": "black", "auto_color": True, "stroke_ratio": 0.08})
        self.assertEqual(text_style({"render": {"font_scale": 5}})["scale"], 2.0)
        self.assertEqual(text_style({"render": {"font_scale": 0.1}})["scale"], 0.5)
        self.assertEqual(text_style({})["scale"], 1.0)
        self.assertEqual(text_style({"render": {"stroke_ratio": -1}})["stroke_ratio"], 0.0)
        self.assertEqual(text_style({"render": {"stroke_ratio": "x"}})["stroke_ratio"], 0.08)

    def test_vertical_style_also_honours_scale_and_color(self):
        from engine.render import render_texts
        box = {"x": 10, "y": 10, "w": 60, "h": 200, "text": "おはよう", "score": 1.0, "vertical": True}
        image = Image.new("RGB", (90, 240), "white")
        render_texts(image, [box], {"おはよう": "早上好"}, {
            "render": {"font_path": "C:/Windows/Fonts/msyh.ttc", "min_font_size": 10,
                       "line_gap": 1.2, "font_scale": 1.0, "color": "#0000ff", "auto_color": False}})
        self.assertTrue(any(b > 150 and r < 100 for r, g, b in image.getdata()))


class VerticalAlignmentTests(unittest.TestCase):
    def test_short_last_column_has_same_top_as_first_column(self):
        from PIL import ImageFont
        from engine.render import draw_vertical
        image = Image.new("RGB", (100, 120), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        box = {"x": 0, "y": 0, "w": 100, "h": 96}
        with patch.object(draw, "text", wraps=draw.text) as text:
            draw_vertical(draw, "田田田田田田", box, font, 20, 1.2)
        calls = text.call_args_list
        self.assertAlmostEqual(calls[0].args[0][1], calls[4].args[0][1])
        self.assertGreater(calls[0].args[0][0], calls[4].args[0][0], "第一列必须在右边")

    def test_actual_vertical_ink_stays_inside_box(self):
        from PIL import ImageFont
        from engine.render import draw_vertical
        image = Image.new("RGB", (100, 140), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        box = {"x": 10, "y": 10, "w": 70, "h": 96}
        draw_vertical(draw, "田田田田田", box, font, 20, 1.2)
        ink = np.any(np.array(image) < 128, axis=2)
        ys, xs = np.where(ink)
        self.assertGreaterEqual(ys.min(), box["y"])
        self.assertLess(ys.max(), box["y"] + box["h"])
        self.assertGreaterEqual(xs.min(), box["x"])
        self.assertLess(xs.max(), box["x"] + box["w"])


class TextReadabilityTests(unittest.TestCase):
    """固定黑字压在深色画面上就是一团糊；靠自动配色 + 描边才看得清。"""

    BOX = {"x": 10, "y": 10, "w": 200, "h": 90, "text": "こんにちは", "score": 1.0, "vertical": False}

    def render(self, background, **render_cfg):
        from engine.render import render_texts
        cfg = {"render": {"font_path": "C:/Windows/Fonts/msyh.ttc", "min_font_size": 10,
                          "line_gap": 1.2, "auto_color": True, "stroke_ratio": 0.08}}
        cfg["render"].update(render_cfg)
        image = Image.new("RGB", (240, 120), background)
        return render_texts(image, [copy.deepcopy(self.BOX)], {"こんにちは": "你好世界"}, cfg)

    @staticmethod
    def brightest(image):
        return max(sum(pixel) for pixel in image.getdata())

    def test_dark_background_gets_light_text(self):
        """深色底上必须用浅字，否则黑字黑底完全看不见。"""
        self.assertGreater(self.brightest(self.render("black")), 600)

    def test_light_background_keeps_dark_text(self):
        image = self.render("white")
        self.assertTrue(any(r < 80 and g < 80 and b < 80 for r, g, b in image.getdata()))

    def test_auto_color_off_uses_fixed_color(self):
        """关掉自动配色后固定色生效，用户想统一风格时用得着。"""
        image = self.render("black", auto_color=False, color="#ff0000")
        self.assertTrue(any(r > 150 and g < 100 and b < 100 for r, g, b in image.getdata()))

    def test_stroke_adds_contrast_ring(self):
        """描边色取字色反色：灰底上深字会被浅色描边包住，浅色像素因此明显变多。"""
        plain = self.render("#808080", stroke_ratio=0)
        stroked = self.render("#808080", stroke_ratio=0.12)
        light = lambda image: sum(1 for p in image.getdata() if sum(p) > 600)
        self.assertGreater(light(stroked), light(plain), "描边应带来一圈与字色相反的像素")

    def test_stroke_zero_is_exactly_off(self):
        from engine.render import stroke_for
        self.assertEqual(stroke_for("#101010", 0.0, 40), (0, None))

    def test_stroke_scales_with_font_size(self):
        from engine.render import stroke_for
        thin, _ = stroke_for("#101010", 0.08, 20)
        thick, _ = stroke_for("#101010", 0.08, 80)
        self.assertLess(thin, thick)
        self.assertGreaterEqual(thin, 1)

    def test_stroke_color_opposes_text_color(self):
        from engine.render import DARK_TEXT, LIGHT_TEXT, stroke_for
        self.assertEqual(stroke_for(DARK_TEXT, 0.08, 40)[1], LIGHT_TEXT)
        self.assertEqual(stroke_for(LIGHT_TEXT, 0.08, 40)[1], DARK_TEXT)

    def test_background_luminance_uses_median(self):
        """抹字没抹干净时残留的原文不该把亮度判断带偏——中位数抗这个。"""
        from engine.render import background_luminance
        image = Image.new("RGB", (240, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([10, 40, 209, 55], fill="black")     # 块内只有一小条是黑的
        self.assertGreater(background_luminance(image, self.BOX), 128)


class LamaWeightTests(unittest.TestCase):
    """LaMa 权重完整性：SHA256 校验 + .ok 标记，防半截文件被 torch 硬加载。"""

    def weight(self, folder, content=b"official-lama-weights"):
        path = os.path.join(folder, "big-lama.pt")
        with open(path, "wb") as handle:
            handle.write(content)
        return path, hashlib.sha256(content).hexdigest()

    def test_missing_file_returns_false_without_error(self):
        from engine.compose import verify_lama_weight
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(verify_lama_weight(os.path.join(folder, "big-lama.pt")))

    def test_verified_file_writes_marker_and_skips_rehash(self):
        from engine import compose
        with tempfile.TemporaryDirectory() as folder:
            path, digest = self.weight(folder)
            with patch.object(compose, "_LAMA_WEIGHT_SHA256", digest):
                self.assertTrue(compose.verify_lama_weight(path))
                with open(path + ".ok", encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle),
                                     {"sha256": digest, "size": os.path.getsize(path)})
                with patch.object(compose, "sha256_file",
                                  side_effect=AssertionError("命中标记不该重算哈希")):
                    self.assertTrue(compose.verify_lama_weight(path))

    def test_corrupt_or_swapped_file_raises_actionable_error(self):
        from engine import compose
        with tempfile.TemporaryDirectory() as folder:
            path, digest = self.weight(folder)
            with patch.object(compose, "_LAMA_WEIGHT_SHA256", digest):
                self.assertTrue(compose.verify_lama_weight(path))
                with open(path, "wb") as handle:      # 半截文件：大小变化使标记失效
                    handle.write(b"half")
                with self.assertRaisesRegex(RuntimeError, "校验失败"):
                    compose.verify_lama_weight(path)

    def test_broken_marker_triggers_rehash(self):
        from engine import compose
        with tempfile.TemporaryDirectory() as folder:
            path, digest = self.weight(folder)
            with open(path + ".ok", "w", encoding="utf-8") as handle:
                handle.write("not json")
            with patch.object(compose, "_LAMA_WEIGHT_SHA256", digest):
                self.assertTrue(compose.verify_lama_weight(path))
                with open(path + ".ok", encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle)["sha256"], digest)

    def test_get_lama_checks_weight_before_loading(self):
        """加载前必须过校验；配置路径不存在时给明确错误；下载完成后补校验。"""
        import sys
        from engine import compose
        with tempfile.TemporaryDirectory() as folder:
            path, digest = self.weight(folder)
            fake_lama = SimpleNamespace(SimpleLama=lambda: "已加载")
            fake_torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: folder))
            cache = {"key": None, "model": None}
            self.addCleanup(os.environ.pop, "LAMA_MODEL", None)
            with patch.dict(sys.modules, {"simple_lama_inpainting": fake_lama,
                                          "torch": fake_torch}), \
                    patch.object(compose, "_LAMA_WEIGHT_SHA256", digest), \
                    patch.object(compose, "_LAMA_CACHE", cache):
                self.assertEqual(compose._get_lama(path), "已加载")
                self.assertEqual(os.environ.get("LAMA_MODEL"), path)
                self.assertTrue(os.path.exists(path + ".ok"))
                with self.assertRaisesRegex(RuntimeError, "不存在"):
                    compose._get_lama(os.path.join(folder, "nope.pt"))
                # 默认路径缺失：构造（下载）完成后必须再校验一次并写标记。
                checkpoint = os.path.join(folder, "checkpoints", "big-lama.pt")
                os.makedirs(os.path.dirname(checkpoint))
                downloads = []

                def fake_build():
                    downloads.append(True)
                    with open(checkpoint, "wb") as handle:
                        handle.write(b"official-lama-weights")
                    return "下载后加载"

                fake_lama.SimpleLama = fake_build
                model = compose._get_lama(None)
                self.assertEqual(model, "下载后加载")
                self.assertEqual(downloads, [True])
                self.assertTrue(os.path.exists(checkpoint + ".ok"))
                self.assertIs(compose._get_lama(None), model, "进程内缓存应命中")


class HotkeyParseTests(unittest.TestCase):
    """快捷键字符串解析：合法的要认，非法/有歧义的要明确拒绝。"""

    def test_common_combinations(self):
        from engine.hotkey import (MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN,
                                   parse_hotkey)
        self.assertEqual(parse_hotkey("ctrl+alt+s"), (MOD_CONTROL | MOD_ALT, 0x53))
        self.assertEqual(parse_hotkey("ctrl+shift+f5"), (MOD_CONTROL | MOD_SHIFT, 0x74))
        self.assertEqual(parse_hotkey("win+1"), (MOD_WIN, 0x31))

    def test_aliases_and_sloppy_input(self):
        """用户手写不该过于较真：大小写、空格、全角加号、control/ctrl 都要认。"""
        from engine.hotkey import parse_hotkey
        canonical = parse_hotkey("ctrl+alt+s")
        for variant in ("  CTRL + Alt + S ", "control+alt+s", "ctrl＋alt＋s",
                        "Ctrl+ALT+s", "ctrl+alt+s+"):
            with self.subTest(variant=variant):
                self.assertEqual(parse_hotkey(variant), canonical)

    def test_rejects_ambiguous_or_empty(self):
        from engine.hotkey import HotkeyError, parse_hotkey
        for bad in ("", "   ", None, "ctrl", "ctrl+alt", "ctrl+a+b",
                    "ctrl+foo", "a", "ctrl+", "ctrl+あ"):
            with self.subTest(bad=bad):
                with self.assertRaises(HotkeyError):
                    parse_hotkey(bad)

    def test_format_is_normalized(self):
        from engine.hotkey import format_hotkey
        self.assertEqual(format_hotkey("ctrl+alt+s"), "Ctrl+Alt+S")
        self.assertEqual(format_hotkey(" ALT+ctrl+p "), "Ctrl+Alt+P")
        self.assertEqual(format_hotkey("ctrl+alt+f1"), "Ctrl+Alt+F1")

    def test_format_rejects_garbage(self):
        from engine.hotkey import HotkeyError, format_hotkey
        with self.assertRaises(HotkeyError):
            format_hotkey("not a hotkey")

    def test_poll_key_allows_bare_keys(self):
        """按住对照用键：允许不带修饰键（Tab/F8），也允许组合键。"""
        from engine.hotkey import MOD_CONTROL, parse_poll_key
        self.assertEqual(parse_poll_key("tab"), (0, 0x09))
        self.assertEqual(parse_poll_key(" F8 "), (0, 0x77))
        self.assertEqual(parse_poll_key("ctrl+tab"), (MOD_CONTROL, 0x09))
        self.assertEqual(parse_poll_key("CTRL＋`"), (MOD_CONTROL, 0xC0))

    def test_poll_key_rejects_empty_and_multi_keys(self):
        from engine.hotkey import HotkeyError, parse_poll_key
        for bad in ("", "   ", None, "ctrl", "ctrl+shift", "a+b", "ctrl+foo"):
            with self.subTest(bad=bad), self.assertRaises(HotkeyError):
                parse_poll_key(bad)

    def test_poll_key_format_is_normalized(self):
        from engine.hotkey import format_poll_key
        self.assertEqual(format_poll_key("tab"), "Tab")
        self.assertEqual(format_poll_key(" ctrl+tab "), "Ctrl+Tab")
        self.assertEqual(format_poll_key("win+f8"), "Win+F8")


class HoldKeyMonitorTests(unittest.TestCase):
    """按住检测：注入假 user32 验证按下/松开沿触发，不碰真实键盘。"""

    class FakeUser32:
        def __init__(self, down=()):
            self.down = set(down)

        def GetAsyncKeyState(self, vk):
            return 0x8000 if vk in self.down else 0

    def monitor(self, key="tab", down=()):
        from engine.hotkey import HoldKeyMonitor
        instance = HoldKeyMonitor(key)
        instance._user32 = self.FakeUser32(down)
        return instance

    def test_invalid_or_empty_key_is_unavailable_and_start_is_noop(self):
        from engine.hotkey import HoldKeyMonitor
        for key in ("", "   ", None, "ctrl+alt", "a+b"):
            with self.subTest(key=key):
                instance = HoldKeyMonitor(key)
                self.assertFalse(instance.available)
                self.assertEqual(instance.label, "")
                instance.start()
                self.assertFalse(instance.timer.isActive())

    def test_press_and_release_edges_fire_once(self):
        monitor = self.monitor("tab")
        self.assertTrue(monitor.available)
        self.assertEqual(monitor.label, "Tab")
        events = []
        monitor.pressed.connect(lambda: events.append("down"))
        monitor.released.connect(lambda: events.append("up"))
        monitor._poll()
        self.assertEqual(events, [])
        monitor._user32.down.add(0x09)
        monitor._poll()
        monitor._poll()
        self.assertEqual(events, ["down"], "按住期间只发一次按下")
        monitor._user32.down.clear()
        monitor._poll()
        monitor._poll()
        self.assertEqual(events, ["down", "up"])

    def test_modifier_must_be_held_with_main_key(self):
        monitor = self.monitor("ctrl+tab", down=(0x09,))
        events = []
        monitor.pressed.connect(lambda: events.append(True))
        monitor._poll()
        self.assertEqual(events, [], "只按主键、没按修饰键时不算触发")
        monitor._user32.down.add(0x11)
        monitor._poll()
        self.assertEqual(events, [True])

    def test_stop_releases_pending_press(self):
        monitor = self.monitor("tab", down=(0x09,))
        releases = []
        monitor.released.connect(lambda: releases.append(True))
        monitor._poll()
        monitor.stop()
        self.assertEqual(releases, [True], "停止时仍按着要补发松开")
        self.assertFalse(monitor.timer.isActive())


class HotkeyConfigTests(unittest.TestCase):
    """config.yaml 的 hotkeys 段读取。"""

    def test_hotkeys_from_config(self):
        from engine.hotkey import hotkeys_from_config
        cfg = {"hotkeys": {"toggle": "ctrl+alt+s", "refresh": "", "settings": None}}
        self.assertEqual(hotkeys_from_config(cfg),
                         {"toggle": "ctrl+alt+s", "refresh": "", "settings": ""})

    def test_missing_section_is_empty(self):
        from engine.hotkey import hotkeys_from_config
        self.assertEqual(hotkeys_from_config({}), {})
        self.assertEqual(hotkeys_from_config(None), {})

    def test_real_config_yaml_is_parseable(self):
        """配置文件里的快捷键必须真的能被解析——写错了要当场发现。"""
        import yaml
        from engine.hotkey import format_hotkey
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.yaml")
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        section = cfg.get("hotkeys") or {}
        self.assertTrue(section, "config.yaml 应带 hotkeys 段")
        for action, text in section.items():
            if not text:
                continue
            with self.subTest(action=action):
                self.assertTrue(format_hotkey(text), f"{action} 的快捷键写错了")

    def test_real_config_yaml_original_key_is_parseable(self):
        """reading.original_key（按住看原图）写错也要当场发现。"""
        import yaml
        from engine.hotkey import format_poll_key
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.yaml")
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        key = (cfg.get("reading") or {}).get("original_key")
        self.assertTrue(key, "config.yaml 的 reading 段应带 original_key")
        self.assertTrue(format_poll_key(key))


class ConfigWritebackTests(unittest.TestCase):
    """快捷键等改动写回 config.yaml：保注释、只动目标键、原子写盘。"""

    SAMPLE = (
        "# 头部注释：不要动\n"
        "reading:\n"
        "  original_key: \"tab\"     # 按住看原图\n"
        "\n"
        "hotkeys:\n"
        "  toggle: \"ctrl+alt+s\"      # 开始 / 暂停\n"
        "  refresh: \"ctrl+alt+2\"     # 刷新\n"
        "\n"
        "# ===== 导出 =====\n"
        "export:\n"
        "  dir: outputs\n"
    )

    def write_sample(self, folder, text=None):
        path = os.path.join(folder, "config.yaml")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text if text is not None else self.SAMPLE)
        return path

    def test_update_keeps_comments_and_other_sections(self):
        import yaml
        from engine.configio import update_section_values
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_sample(folder)
            with open(path, encoding="utf-8") as handle:
                top_keys_before = list(yaml.safe_load(handle))
            changed = update_section_values(path, "hotkeys", {"toggle": "ctrl+alt+f1"})
            self.assertEqual(changed, ["toggle"])
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn('  toggle: "ctrl+alt+f1"      # 开始 / 暂停', text, "行内注释必须保留")
            self.assertIn('  refresh: "ctrl+alt+2"     # 刷新', text, "未涉及的键不能动")
            self.assertIn("# 头部注释：不要动", text)
            self.assertIn("  original_key: \"tab\"     # 按住看原图", text)
            self.assertIn("# ===== 导出 =====", text)
            cfg = yaml.safe_load(text)
            self.assertEqual(cfg["hotkeys"]["toggle"], "ctrl+alt+f1")
            self.assertEqual(cfg["hotkeys"]["refresh"], "ctrl+alt+2")
            self.assertEqual(cfg["export"]["dir"], "outputs")
            self.assertEqual(list(cfg), top_keys_before, "不能增删或重排顶层段落")
            self.assertFalse(os.path.exists(path + ".tmp"), "不能留下临时文件")

    def test_clearing_hotkey_writes_empty_string(self):
        import yaml
        from engine.configio import update_section_values
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_sample(folder)
            update_section_values(path, "hotkeys", {"refresh": ""})
            with open(path, encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
            self.assertEqual(cfg["hotkeys"]["refresh"], "", "清空的键也要写回，重启后才一致")

    def test_missing_key_is_inserted_inside_section(self):
        import yaml
        from engine.configio import update_section_values
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_sample(folder)
            update_section_values(path, "hotkeys", {"leave": "ctrl+alt+4"})
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertEqual(yaml.safe_load(text)["hotkeys"]["leave"], "ctrl+alt+4")
            self.assertLess(text.index("leave:"), text.index("# ===== 导出 ====="),
                            "新键必须插在 hotkeys 段内，不能跑到别的段后面")

    def test_missing_section_is_created_at_end(self):
        import yaml
        from engine.configio import update_section_values
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_sample(folder, "export:\n  dir: outputs\n")
            update_section_values(path, "hotkeys", {"toggle": "ctrl+alt+s"})
            with open(path, encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
            self.assertEqual(cfg["hotkeys"], {"toggle": "ctrl+alt+s"})
            self.assertEqual(cfg["export"]["dir"], "outputs")

    def test_identical_values_leave_file_untouched(self):
        from engine.configio import update_section_values
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_sample(folder)
            self.assertEqual(update_section_values(path, "hotkeys", {"toggle": "ctrl+alt+s"}), [])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), self.SAMPLE, "值没变就不该重写文件")


class HotkeyManagerTests(unittest.TestCase):
    """管理器行为：无绑定不启线程、停止不卡住、失败不抛。"""

    def test_no_bindings_does_not_start_thread(self):
        from engine.hotkey import HotkeyManager
        manager = HotkeyManager({"toggle": "", "refresh": None})
        manager.start()
        self.assertFalse(manager.enabled)
        self.assertEqual(manager.registered, {})
        manager.stop()

    def test_start_and_stop_do_not_hang(self):
        """消息泵会阻塞在 GetMessageW，start/stop 都不能被它卡住。"""
        import time
        from engine.hotkey import HotkeyManager
        manager = HotkeyManager({"toggle": "ctrl+alt+f9"})
        started = time.time()
        manager.start()
        self.assertLess(time.time() - started, 3.0)
        stopped = time.time()
        manager.stop()
        self.assertLess(time.time() - stopped, 2.0)
        self.assertFalse(manager.enabled)

    def test_invalid_hotkey_reported_not_raised(self):
        from engine.hotkey import HotkeyManager
        manager = HotkeyManager({"toggle": "ctrl+alt"})
        messages = []
        manager.message.connect(messages.append)
        manager.start()          # 不能抛异常
        self.assertIn("toggle", manager.failures)
        manager.stop()

    def test_stop_is_idempotent(self):
        from engine.hotkey import HotkeyManager
        manager = HotkeyManager({"toggle": "ctrl+alt+f8"})
        manager.start()
        manager.stop()
        manager.stop()           # 再停一次不能崩
        self.assertFalse(manager.enabled)


class HotkeyWiringTests(unittest.TestCase):
    """热键动作 → 控制器方法 的接线与状态守卫。"""

    def controller(self):
        from ui.bridge import Bridge
        from ui.reading import NativePanel
        from ui.realtime import RealtimeController
        cfg = {"llm": {"provider": "mock"}, "hotkeys": {"toggle": "ctrl+alt+s"}}
        panel = NativePanel(cfg)
        bridge = Bridge(cfg, panel=panel)
        return RealtimeController(bridge, panel, cfg), bridge

    def test_action_table_covers_all_config_actions(self):
        """config.yaml 里出现的每个动作都要有对应的控制器方法。"""
        import yaml
        from ui.realtime import RealtimeController
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.yaml")
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        ctrl, _ = self.controller()
        for action in (cfg.get("hotkeys") or {}):
            with self.subTest(action=action):
                self.assertIn(action, ctrl.HOTKEY_ACTIONS)
        ctrl.hotkeys.stop()

    def test_guards_block_unready_and_busy(self):
        ctrl, bridge = self.controller()
        calls = []
        ctrl.toggle = lambda: calls.append("toggle")
        # 模型未就绪：不能触发
        ctrl._on_hotkey("toggle")
        self.assertEqual(calls, [])
        # 就绪后放行
        bridge._pipeline = object()
        ctrl._on_hotkey("toggle")
        self.assertEqual(calls, ["toggle"])
        # 正忙：再次被拦
        bridge._busy = True
        ctrl._on_hotkey("toggle")
        self.assertEqual(calls, ["toggle"])
        ctrl.hotkeys.stop()

    def test_picking_blocks_every_action(self):
        ctrl, bridge = self.controller()
        bridge._pipeline = object()
        bridge._picking = True
        calls = []
        for action in ctrl.HOTKEY_ACTIONS:
            setattr(ctrl, ctrl.HOTKEY_ACTIONS[action], lambda a=action: calls.append(a))
        for action in ctrl.HOTKEY_ACTIONS:
            ctrl._on_hotkey(action)
        self.assertEqual(calls, [])
        ctrl.hotkeys.stop()

    def test_unknown_action_is_ignored(self):
        ctrl, _ = self.controller()
        ctrl._on_hotkey("does_not_exist")   # 不能抛
        ctrl.hotkeys.stop()

    def test_apply_hotkeys_updates_config(self):
        ctrl, bridge = self.controller()
        ctrl.apply_hotkeys({"toggle": "ctrl+alt+f1", "refresh": ""})
        self.assertEqual(ctrl.cfg["hotkeys"], {"toggle": "ctrl+alt+f1", "refresh": ""})
        ctrl.hotkeys.stop()

    def test_apply_hotkeys_writes_back_to_config_file(self):
        """带 _config_path 时改绑要真的落盘；用临时副本，不碰真配置。"""
        import yaml
        from ui.bridge import Bridge
        from ui.reading import NativePanel
        from ui.realtime import RealtimeController
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "config.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('hotkeys:\n'
                             '  toggle: "ctrl+alt+s"   # 开始\n'
                             '  refresh: "ctrl+alt+2"  # 刷新\n')
            cfg = {"llm": {"provider": "mock"},
                   "hotkeys": {"toggle": "ctrl+alt+s", "refresh": "ctrl+alt+2"},
                   "_config_path": path}
            panel = NativePanel(cfg)
            bridge = Bridge(cfg, panel=panel)
            ctrl = RealtimeController(bridge, panel, cfg)
            ctrl.apply_hotkeys({"toggle": "ctrl+alt+f1", "refresh": ""})
            ctrl.hotkeys.stop()
            with open(path, encoding="utf-8") as handle:
                saved = yaml.safe_load(handle)
            self.assertEqual(saved["hotkeys"]["toggle"], "ctrl+alt+f1")
            self.assertEqual(saved["hotkeys"]["refresh"], "")
            with open(path, encoding="utf-8") as handle:
                self.assertIn("# 开始", handle.read(), "行内注释不能被抹掉")

    def test_toolbar_tooltips_show_hotkeys(self):
        from ui.reading import ReadingToolbar
        cfg = {"hotkeys": {"toggle": "ctrl+alt+s", "reselect": "ctrl+alt+1"}}
        bar = ReadingToolbar(cfg)
        self.assertIn("Ctrl+Alt+S", bar.toggle_button.toolTip())
        self.assertIn("Ctrl+Alt+1", bar.region_button.toolTip())

    def test_toolbar_tooltips_work_without_hotkeys(self):
        from ui.reading import ReadingToolbar
        bar = ReadingToolbar({"hotkeys": {"toggle": ""}})
        self.assertEqual(bar.toggle_button.toolTip(), "开始或暂停透明实时阅读")


class ExportRenderTests(unittest.TestCase):
    """对照文本渲染：格式、去噪、顺序、边界。"""

    def test_noise_detection(self):
        from engine.export import is_noise
        for text in ("", "   ", "。", "…", "※→", "\u3000"):
            with self.subTest(text=text):
                self.assertTrue(is_noise(text), f"{text!r} 应判为符号碎片")
        for text in ("こんにちは", "東堂容太", "5.7.6", "Hello"):
            with self.subTest(text=text):
                self.assertFalse(is_noise(text), f"{text!r} 不应判为符号碎片")

    def test_clean_pairs_keeps_first_seen_order(self):
        from engine.export import clean_pairs
        pairs = [("B", "乙"), ("A", "甲"), ("C", "丙")]
        self.assertEqual(clean_pairs(pairs), [("B", "乙"), ("A", "甲"), ("C", "丙")])

    def test_clean_pairs_drops_empty_and_noise(self):
        from engine.export import clean_pairs
        pairs = [("A", "甲"), ("", "空原文"), ("B", ""), ("…", "。"), ("C", "丙")]
        self.assertEqual(clean_pairs(pairs), [("A", "甲"), ("C", "丙")])

    def test_clean_pairs_dedupes_by_source(self):
        from engine.export import clean_pairs
        pairs = [("A", "甲"), ("B", "乙"), ("A", "第二次应丢")]
        self.assertEqual(clean_pairs(pairs), [("A", "甲"), ("B", "乙")])

    def test_clean_pairs_accepts_dict_and_list_shapes(self):
        from engine.export import clean_pairs
        self.assertEqual(clean_pairs({"A": "甲"}), [("A", "甲")])
        self.assertEqual(clean_pairs([{"source": "A", "translation": "甲"}]), [("A", "甲")])
        self.assertEqual(clean_pairs([{"src": "A", "dst": "甲"}]), [("A", "甲")])
        # 乱七八糟的输入不能让它崩，只能被忽略
        self.assertEqual(clean_pairs(["垃圾", ("只有一个",), 42, None]), [])

    def test_clean_pairs_handles_none_and_empty(self):
        from engine.export import clean_pairs
        for bad in (None, {}, [], (), "字符串", 123):
            with self.subTest(bad=bad):
                self.assertEqual(clean_pairs(bad), [])

    def test_render_format_source_then_target(self):
        from engine.export import render_pairs
        text = render_pairs({"こんにちは": "你好", "さようなら": "再见"})
        self.assertEqual(text, "こんにちは\n你好\n\nさようなら\n再见\n")

    def test_render_empty_returns_empty_string(self):
        """没内容要返回空串，调用方才能区分「没东西可导」和「导出成功但内容为空」。"""
        from engine.export import render_pairs
        self.assertEqual(render_pairs({}), "")
        self.assertEqual(render_pairs({"…": "。"}), "")

    def test_render_with_header(self):
        from engine.export import render_pairs
        text = render_pairs({"A": "甲"}, header="来源：当前页")
        self.assertTrue(text.startswith("来源：当前页\n\n"))
        self.assertIn("A\n甲", text)

    def test_render_pages_skips_empty_pages(self):
        from engine.export import render_pages
        text = render_pages([("第 1 页", {"A": "甲"}),
                             ("第 2 页", {"…": "。"}),
                             ("第 3 页", {"B": "乙"})])
        self.assertIn("===== 第 1 页 =====", text)
        self.assertNotIn("第 2 页", text, "空页应被跳过")
        self.assertIn("===== 第 3 页 =====", text)

    def test_render_pages_all_empty(self):
        from engine.export import render_pages
        self.assertEqual(render_pages([("第 1 页", {})]), "")


class ExportFileTests(unittest.TestCase):
    """写文件：命名、不覆盖、失败要抛。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fanyi_export_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def config(self, **overrides):
        export_cfg = {"dir": self.tmp, "drop_noise": True, "timestamp_name": True}
        export_cfg.update(overrides)
        return {"export": export_cfg}

    def test_write_creates_file_with_utf8_content(self):
        from ui.bridge import write_export
        path, count = write_export({"こんにちは": "你好"}, self.config())
        self.assertEqual(count, 1)
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("こんにちは", content)
        self.assertIn("你好", content)

    def test_write_rejects_empty_content(self):
        from ui.bridge import write_export
        with self.assertRaises(ValueError):
            write_export({}, self.config())
        with self.assertRaises(ValueError):
            write_export({"…": "。"}, self.config())

    def test_repeated_exports_do_not_overwrite(self):
        """同一秒内连点两次导出不能把前一份盖掉——那是静默的数据丢失。"""
        import datetime
        from ui.bridge import write_export
        moment = datetime.datetime(2026, 10, 11, 16, 0, 2)
        first, _ = write_export({"A": "甲"}, self.config(), when=moment)
        second, _ = write_export({"B": "乙"}, self.config(), when=moment)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(first), "第一份导出不能被覆盖")
        with open(first, encoding="utf-8") as handle:
            self.assertIn("甲", handle.read())

    def test_creates_missing_directory(self):
        from ui.bridge import write_export
        nested = os.path.join(self.tmp, "a", "b")
        path, _ = write_export({"A": "甲"}, self.config(dir=nested))
        self.assertTrue(os.path.isdir(nested))
        self.assertTrue(os.path.exists(path))

    def test_drop_noise_can_be_disabled(self):
        from ui.bridge import write_export
        path, count = write_export({"…": "……"}, self.config(drop_noise=False))
        self.assertEqual(count, 1)

    def test_fixed_name_when_timestamp_disabled(self):
        from ui.bridge import write_export
        path, _ = write_export({"A": "甲"}, self.config(timestamp_name=False))
        self.assertEqual(os.path.basename(path), "译文对照.txt")


class MarkingOverlayTests(unittest.TestCase):
    """覆盖层标记：句框记录、点击命中、标记模式切换。"""

    def overlay(self):
        from ui.reading import ReadingOverlay
        overlay = ReadingOverlay({"reading": {}})
        overlay.resize(200, 100)
        overlay._image_size = QImage(200, 100, QImage.Format_RGB32).size()
        return overlay

    def test_capture_sentences_skips_invalid(self):
        overlay = self.overlay()
        overlay._capture_sentences([
            {"x": 0, "y": 0, "w": 20, "h": 10, "text": "句一"},
            {"x": 0, "y": 0, "w": 20, "h": 10, "text": "  "},     # 空文本
            {"x": 0, "y": 0, "w": 0, "h": 10, "text": "零宽"},     # 宽为 0
            {"text": "缺坐标"},
            "字符串",
        ])
        self.assertEqual([s[4] for s in overlay._sentences], ["句一"])

    def test_sentence_at_maps_point_and_prefers_smallest(self):
        from PySide6.QtCore import QPointF
        overlay = self.overlay()
        overlay._sentences = [(0, 0, 100, 40, "大框"), (10, 10, 20, 10, "小框")]
        self.assertEqual(overlay.sentence_at(QPointF(15, 15)), "小框", "嵌套时要点中最具体的框")
        self.assertEqual(overlay.sentence_at(QPointF(90, 35)), "大框")
        self.assertEqual(overlay.sentence_at(QPointF(150, 90)), "")

    def test_sentence_at_without_result_is_empty(self):
        from PySide6.QtCore import QPointF
        overlay = self.overlay()
        self.assertEqual(overlay.sentence_at(QPointF(1, 1)), "")

    def test_marking_toggles_input_transparency(self):
        overlay = self.overlay()
        self.assertTrue(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
        overlay.set_marking(True)
        self.assertFalse(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
        self.assertTrue(overlay._mark_mode)
        overlay.set_marking(False)
        self.assertTrue(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
        self.assertFalse(overlay._mark_mode)

    def test_mouse_press_emits_sentence_only_in_mark_mode(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent
        overlay = self.overlay()
        overlay._sentences = [(0, 0, 50, 50, "句一")]
        seen = []
        overlay.sentence_clicked.connect(seen.append)
        event = QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(10, 10),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        overlay.mousePressEvent(event)          # 非标记模式：不触发，保持点击穿透
        self.assertEqual(seen, [])
        overlay.set_marking(True)
        overlay.mousePressEvent(event)
        self.assertEqual(seen, ["句一"])

    def test_set_marks_normalizes_and_dedupes(self):
        overlay = self.overlay()
        overlay.set_marks(["句一", "句一", "  ", "句二"])
        self.assertEqual(overlay._marks, {"句一", "句二"})

    def test_fade_in_runs_then_settles(self):
        overlay = self.overlay()
        overlay._anim_total = 100
        overlay._anim_elapsed = 0
        overlay._fade = 0.0
        overlay._anim_timer.start()
        overlay._on_anim_tick()
        self.assertGreater(overlay._fade, 0.0)
        self.assertLess(overlay._fade, 1.0)
        for _ in range(20):
            overlay._on_anim_tick()
        self.assertEqual(overlay._fade, 1.0)
        self.assertFalse(overlay._anim_timer.isActive(), "淡入结束后要停表，不能空转")

    def test_fade_disabled_shows_immediately(self):
        from ui.reading import ReadingOverlay
        overlay = ReadingOverlay({"reading": {"fade_in": False}})
        overlay._fade = 0.5
        overlay._start_fade()
        self.assertEqual(overlay._fade, 1.0)
        self.assertFalse(overlay._anim_timer.isActive())

    def test_set_result_starts_fade_and_records_sentences(self):
        from PySide6.QtCore import QBuffer, QIODevice
        overlay = self.overlay()
        image = QImage(40, 20, QImage.Format_RGB32)
        image.fill(Qt.red)
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        image.save(buffer, "PNG")
        payload = {
            "ok": True,
            "image": "data:image/png;base64," + base64.b64encode(bytes(buffer.data())).decode(),
            "boxes": [{"x": 0, "y": 0, "w": 10, "h": 10, "text": "句一"}],
        }
        overlay.set_result(payload)
        self.assertEqual([s[4] for s in overlay._sentences], ["句一"])
        self.assertEqual(overlay._fade, 0.0)
        self.assertTrue(overlay._anim_timer.isActive())

    def test_peeking_toggles_input_and_is_exclusive(self):
        overlay = self.overlay()
        overlay.set_peeking(True)
        self.assertTrue(overlay._peek_mode)
        self.assertFalse(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
        overlay.set_marking(True)
        self.assertFalse(overlay._peek_mode, "开标记应关掉单句对照")
        self.assertTrue(overlay._mark_mode)
        overlay.set_peeking(True)
        self.assertFalse(overlay._mark_mode, "开单句对照应关掉标记")
        overlay.set_peeking(False)
        self.assertTrue(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
        self.assertIsNone(overlay._peek_rect)

    def test_peek_click_toggles_rect(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent
        overlay = self.overlay()
        overlay._sentences = [(0, 0, 50, 50, "句一")]
        overlay.set_peeking(True)
        event = QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(10, 10),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        overlay.mousePressEvent(event)
        self.assertEqual(overlay._peek_rect, (0.0, 0.0, 50.0, 50.0))
        overlay.mousePressEvent(event)          # 再点同一句收起
        self.assertIsNone(overlay._peek_rect)

    def test_box_at_returns_none_on_miss(self):
        from PySide6.QtCore import QPointF
        overlay = self.overlay()
        overlay._sentences = [(0, 0, 10, 10, "句一")]
        self.assertEqual(overlay.box_at(QPointF(5, 5)), (0.0, 0.0, 10.0, 10.0))
        self.assertIsNone(overlay.box_at(QPointF(90, 90)))

    def test_clear_result_clears_peek(self):
        overlay = self.overlay()
        overlay._peek_rect = (0, 0, 10, 10)
        overlay.clear_result()
        self.assertIsNone(overlay._peek_rect)


class ExportMarkTests(unittest.TestCase):
    """译错标记：导出文本带上标记行与总数，理由按原文匹配。"""

    def test_mark_reason_accepts_dict_and_set(self):
        from engine.export import mark_reason
        self.assertEqual(mark_reason({"A": "漏译"}, "A"), "漏译")
        self.assertEqual(mark_reason({}, "A"), "")
        self.assertEqual(mark_reason({"A": "译得不好"}, "B"), "")
        self.assertEqual(mark_reason({"A"}, "A"), "译得不好")
        self.assertEqual(mark_reason(["A"], "A"), "译得不好")
        self.assertEqual(mark_reason({"A"}, "B"), "")
        self.assertEqual(mark_reason(None, "A"), "")

    def test_render_pairs_appends_marker_and_header_count(self):
        from engine.export import render_pairs
        text = render_pairs([("A", "甲"), ("B", "乙")],
                            header="本次阅读", marks={"A": "译得不好"})
        self.assertIn("本次阅读（1 句已标记待复核）", text)
        self.assertIn("A\n甲\n  [!] 标记：译得不好", text)
        self.assertIn("B\n乙", text)
        self.assertNotIn("B\n乙\n  [!]", text)

    def test_render_pairs_without_matching_mark_has_no_marker(self):
        from engine.export import render_pairs
        text = render_pairs([("A", "甲")], header="x", marks={"B": "译得不好"})
        self.assertNotIn("[!]", text)
        self.assertEqual(text, "x\n\nA\n甲\n")

    def test_render_pages_marks_apply_across_pages(self):
        from engine.export import render_pages
        pages = [("第一页", [("A", "甲")]), ("第二页", [("B", "乙")])]
        text = render_pages(pages, marks={"A": "译得不好"})
        self.assertEqual(text.count("[!] 标记：译得不好"), 1)
        self.assertIn("===== 第一页 =====", text)
        self.assertIn("===== 第二页 =====", text)


class ExportWiringTests(unittest.TestCase):
    """导出接线：控制器累积、UI 信号、异常路径。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fanyi_export_wire_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def controller(self):
        from ui.bridge import Bridge
        from ui.reading import NativePanel
        from ui.realtime import RealtimeController
        cfg = {"llm": {"provider": "mock"}, "hotkeys": {},
               "export": {"dir": self.tmp, "drop_noise": True, "timestamp_name": True}}
        panel = NativePanel(cfg)
        bridge = Bridge(cfg, panel=panel)
        return RealtimeController(bridge, panel, cfg), bridge, cfg

    def test_session_accumulates_across_pages(self):
        ctrl, _, _ = self.controller()
        ctrl._present({"pairs": {"句一": "译一", "句二": "译二"}}, 1)
        ctrl._present({"pairs": {"句三": "译三"}}, 2)
        self.assertEqual(ctrl.session_pairs,
                         {"句一": "译一", "句二": "译二", "句三": "译三"})
        ctrl.hotkeys.stop()

    def test_session_dedupes_repeated_source(self):
        ctrl, _, _ = self.controller()
        ctrl._present({"pairs": {"句一": "译一"}}, 1)
        ctrl._present({"pairs": {"句一": "第二次应丢"}}, 2)
        self.assertEqual(ctrl.session_pairs, {"句一": "译一"})
        ctrl.hotkeys.stop()

    def test_export_session_writes_accumulated(self):
        ctrl, _, _ = self.controller()
        ctrl._present({"pairs": {"句一": "译一"}}, 1)
        logs = []
        ctrl.bridge.log_line.connect(logs.append)
        ctrl.export_session()
        joined = "\n".join(logs)
        self.assertIn("已导出", joined)
        written = [f for f in os.listdir(self.tmp) if f.endswith(".txt")]
        self.assertEqual(len(written), 1)
        ctrl.hotkeys.stop()

    def test_export_session_without_content_says_so(self):
        ctrl, _, _ = self.controller()
        logs = []
        ctrl.bridge.log_line.connect(logs.append)
        ctrl.export_session()
        self.assertTrue(any("还没有可导出的译文" in m for m in logs))
        self.assertEqual(os.listdir(self.tmp), [])
        ctrl.hotkeys.stop()

    def test_export_falls_back_to_last_page(self):
        """没累积过（比如只做了单页翻译）也要能导出最近一页。"""
        ctrl, _, _ = self.controller()
        ctrl._last_payload = {"pairs": {"单页句": "单页译"}}
        logs = []
        ctrl.bridge.log_line.connect(logs.append)
        ctrl.export_session()
        self.assertTrue(any("已导出" in m for m in logs))
        ctrl.hotkeys.stop()

    def test_export_while_running_is_refused(self):
        ctrl, _, _ = self.controller()
        ctrl._present({"pairs": {"句一": "译一"}}, 1)
        ctrl.running = True
        logs = []
        ctrl.bridge.log_line.connect(logs.append)
        ctrl.export_session()
        self.assertTrue(any("先暂停再导出" in m for m in logs))
        self.assertEqual(os.listdir(self.tmp), [], "不应写出半截内容")
        ctrl.running = False
        ctrl.hotkeys.stop()

    def test_remember_pairs_survives_odd_shapes(self):
        ctrl, _, _ = self.controller()
        for weird in (None, [], "字符串", 123, {"pairs": None}, {"pairs": 42}):
            with self.subTest(weird=weird):
                ctrl._remember_pairs(weird)
        self.assertEqual(ctrl.session_pairs, {})
        ctrl.hotkeys.stop()

    def test_bad_export_dir_reports_oserror(self):
        """目录不可写要明确报错，不能崩，也不能假装成功。"""
        from ui.bridge import Bridge
        ctrl, _, cfg = self.controller()
        cfg["export"]["dir"] = "Z:/definitely_not_a_drive/foo"
        logs = []
        ctrl.bridge.log_line.connect(logs.append)
        ctrl.bridge.export_pairs({"A": "甲"}, "测试")
        self.assertTrue(any("导出失败" in m for m in logs))
        ctrl.hotkeys.stop()

    def test_ui_signals_and_buttons_exist(self):
        from ui.reading import ReadingToolbar, NativePanel
        ctrl, bridge, cfg = self.controller()
        self.assertTrue(hasattr(ctrl.panel, "export_button"))
        self.assertTrue(hasattr(ctrl.panel, "export_requested"))
        self.assertTrue(hasattr(ctrl.toolbar, "export_button"))
        self.assertTrue(hasattr(ctrl.toolbar, "export_requested"))
        self.assertTrue(hasattr(ctrl.toolbar, "mark_button"))
        self.assertTrue(hasattr(ctrl.toolbar, "mark_requested"))
        self.assertTrue(hasattr(ctrl.toolbar, "peek_button"))
        self.assertTrue(hasattr(ctrl.toolbar, "peek_requested"))
        self.assertTrue(hasattr(ctrl.overlay, "sentence_clicked"))
        ctrl.hotkeys.stop()

    def test_toolbar_export_signal_reaches_controller(self):
        """工具栏「导出」按钮必须已经接到控制器的 export_session（在 __init__ 里接的）。"""
        ctrl, _, _ = self.controller()
        calls = []
        ctrl.export_session = lambda: calls.append(True)
        ctrl.toolbar.export_button.click()
        self.assertEqual(calls, [True], "工具栏导出按钮没有连到控制器")
        ctrl.hotkeys.stop()

    def test_click_cycles_mark(self):
        """同一句再点一次即取消标记。"""
        ctrl, _, _ = self.controller()
        ctrl._on_sentence_clicked("句一")
        self.assertEqual(ctrl.session_marks, {"句一": "译得不好"})
        ctrl._on_sentence_clicked("句一")
        self.assertEqual(ctrl.session_marks, {})
        ctrl.hotkeys.stop()

    def test_empty_sentence_is_ignored(self):
        ctrl, _, _ = self.controller()
        ctrl._on_sentence_clicked("   ")
        ctrl._on_sentence_clicked(None)
        self.assertEqual(ctrl.session_marks, {})
        ctrl.hotkeys.stop()

    def test_export_includes_marked_sentence(self):
        ctrl, _, _ = self.controller()
        ctrl._present({"pairs": {"句一": "译一", "句二": "译二"}}, 1)
        ctrl._on_sentence_clicked("句一")
        ctrl.export_session()
        written = [f for f in os.listdir(self.tmp) if f.endswith(".txt")]
        self.assertEqual(len(written), 1)
        with open(os.path.join(self.tmp, written[0]), encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("[!] 标记：译得不好", text)
        self.assertIn("1 句已标记待复核", text)
        ctrl.hotkeys.stop()

    def test_toolbar_mark_button_wires_to_overlay(self):
        ctrl, _, _ = self.controller()
        ctrl.toolbar.mark_button.setChecked(True)
        self.assertTrue(ctrl.overlay._mark_mode, "开标记模式后覆盖层应接收点击")
        ctrl.toolbar.mark_button.setChecked(False)
        self.assertFalse(ctrl.overlay._mark_mode)
        ctrl.hotkeys.stop()

    def test_toolbar_peek_button_wires_and_is_exclusive(self):
        ctrl, _, _ = self.controller()
        ctrl.toolbar.peek_button.setChecked(True)
        self.assertTrue(ctrl.overlay._peek_mode)
        # 开标记应自动关掉单句对照，两者互斥
        ctrl.toolbar.mark_button.setChecked(True)
        self.assertFalse(ctrl.overlay._peek_mode)
        self.assertTrue(ctrl.overlay._mark_mode)
        self.assertFalse(ctrl.toolbar.peek_button.isChecked())
        ctrl.toolbar.mark_button.setChecked(False)
        ctrl.hotkeys.stop()

    def test_pause_keeps_marks_for_later_export(self):
        ctrl, _, _ = self.controller()
        ctrl._on_sentence_clicked("句一")
        ctrl.pause()
        self.assertEqual(ctrl.session_marks, {"句一": "译得不好"}, "暂停不应丢掉标记")
        ctrl.hotkeys.stop()


class MetaResponseTests(unittest.TestCase):
    """模型不翻译、改去"评论"输入时，必须判无译文而不是把评论盖进漫画。

    实测 qwen2.5:7b 拿到 OCR 垃圾（单个字母 "n"）会回一句「无法辨认」——那是在
    复述系统提示词，却被当成译文抹掉原文盖了上去。
    """

    def test_chinese_meta_responses_are_rejected(self):
        for text in ("无法辨认", "无法识别", "无法翻译", "无法确定",
                     "请提供原文", "未提供原文", "原文缺失", "辨认不出"):
            with self.subTest(text=text):
                self.assertTrue(is_meta_response(text))
                self.assertEqual(sanitize_translation(text), "")

    def test_english_meta_responses_are_rejected(self):
        for text in ("I cannot translate this.", "Unable to translate",
                     "Please provide the text", "As an AI language model"):
            with self.subTest(text=text):
                self.assertEqual(sanitize_translation(text), "")

    def test_normal_translations_survive(self):
        for text in ("你好", "你喜欢宫原吧", "没事！", "但很快就要熄灯了吧？",
                     "单恋……", "请提供帮助"):
            with self.subTest(text=text):
                self.assertFalse(is_meta_response(text))
                self.assertEqual(sanitize_translation(text), text)

    def test_long_text_with_marker_is_kept(self):
        """正常对白里偶尔出现"请提供"，不能因为命中关键词就把整句丢掉。"""
        text = "他说：请提供文件，我明天带过来给你看"
        self.assertFalse(is_meta_response(text))
        self.assertEqual(sanitize_translation(text), text)

    def test_empty_input_is_not_meta(self):
        self.assertFalse(is_meta_response(""))
        self.assertFalse(is_meta_response(None))


class LowConfidenceFilterTests(unittest.TestCase):
    """置信度过低的块要丢弃并保留原文，别送去翻译换回一句垃圾。"""

    def box(self, text, score):
        return {"x": 1, "y": 1, "w": 10, "h": 20, "text": text, "score": score,
                "vertical": True}

    def test_drops_only_below_threshold(self):
        boxes = [self.box("正常对白", 0.95), self.box("n", 0.20),
                 self.box("D·", 0.50), self.box("あん", 0.70)]
        kept = filter_low_confidence(boxes, 0.6)
        self.assertEqual([b["text"] for b in kept], ["正常对白", "あん"])

    def test_threshold_zero_disables_filtering(self):
        boxes = [self.box("n", 0.20), self.box("D·", 0.05)]
        self.assertEqual(len(filter_low_confidence(boxes, 0)), 2)

    def test_bad_threshold_is_ignored(self):
        boxes = [self.box("n", 0.20)]
        for bad in (None, "", "abc", float("nan")):
            with self.subTest(bad=bad):
                self.assertEqual(len(filter_low_confidence(boxes, bad)), 1)

    def test_missing_score_is_kept(self):
        """没有 score 字段（老数据/外部传入）不能当成 0 丢掉。"""
        self.assertEqual(len(filter_low_confidence([{"text": "x"}], 0.6)), 1)

    def test_default_threshold_separates_real_page_scores(self):
        """真实漫画页实测：0.11~0.50 是垃圾，0.70+ 是正常对白。"""
        from engine.ocr import DEFAULT_MIN_SCORE
        garbage = [self.box(t, s) for t, s in
                   (("n", 0.20), ("S-我へi", 0.37), ("D·", 0.50), ("□", 0.19), ("中", 0.22))]
        real = [self.box(t, s) for t, s in
                (("あん", 0.70), ("おま…はっ！？", 0.70), ("大丈夫ぅ！", 1.00))]
        self.assertEqual(filter_low_confidence(garbage, DEFAULT_MIN_SCORE), [])
        self.assertEqual(filter_low_confidence(real, DEFAULT_MIN_SCORE), real)


class OcrMinScoreConfigTests(unittest.TestCase):
    """ocr.min_score 要能配；缺省用 0.6。"""

    def test_engine_reads_min_score_from_config(self):
        from engine.ocr import OcrEngine, DEFAULT_MIN_SCORE
        engine = OcrEngine.__new__(OcrEngine)
        for raw, want in ((0.4, 0.4), (0, 0), (None, DEFAULT_MIN_SCORE),
                          ("bad", DEFAULT_MIN_SCORE), (1.5, 1.5)):
            with self.subTest(raw=raw):
                cfg = {"ocr": {"min_score": raw}}
                ocr_cfg = cfg.get("ocr", {})
                from engine.ocr import _number
                self.assertEqual(_number(ocr_cfg.get("min_score"), DEFAULT_MIN_SCORE), want)

    def test_recognize_filters_and_reports(self):
        from engine.pipeline import Pipeline

        class FakeOcr:
            min_score = 0.6

            def detect(self, image):
                return [{"x": 5, "y": 5, "w": 30, "h": 10, "text": "こんにちは",
                         "score": 0.95, "vertical": False},
                        {"x": 5, "y": 20, "w": 10, "h": 10, "text": "n",
                         "score": 0.20, "vertical": True}]

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.cfg = copy.deepcopy(CFG)
        pipeline.ocr = FakeOcr()
        pipeline.translator = Translator(copy.deepcopy(CFG))
        logs = []
        boxes, _ = pipeline.recognize(np.full((40, 60, 3), 200, np.uint8),
                                      progress=logs.append)
        self.assertEqual([b["text"] for b in boxes], ["こんにちは"])
        self.assertTrue(any("丢弃 1 个置信度过低" in s for s in logs), logs)

    def test_recognize_without_filter_keeps_all(self):
        from engine.pipeline import Pipeline

        class FakeOcr:
            min_score = 0

            def detect(self, image):
                return [{"x": 5, "y": 5, "w": 10, "h": 20, "text": "n",
                         "score": 0.20, "vertical": True}]

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.cfg = copy.deepcopy(CFG)
        pipeline.ocr = FakeOcr()
        pipeline.translator = Translator(copy.deepcopy(CFG))
        boxes, _ = pipeline.recognize(np.full((40, 60, 3), 200, np.uint8))
        self.assertEqual([b["text"] for b in boxes], ["n"])


class RealPageMergeTests(unittest.TestCase):
    """真实漫画页（インゴシマ）实测的 7 组同气泡列，坐标都是实测值。

    旧规则用「框宽比值 ≤1.5」和「纵向溢出 ≤0.6 字宽」判定，7 组全被拒——
    因为 PaddleOCR 常把 2~3 列并进一个框，框宽天然差一倍。结果一个句子被
    拆成两半分别翻译（"幼馴染で" 单独译成"小学同学"）。
    新规则改看字号（sqrt(w×h/字数)）与对齐方式，7 组全部合并。
    """

    def col(self, text, x, y, w, h):
        return {"x": x, "y": y, "w": w, "h": h, "text": text,
                "score": 0.95, "vertical": True}

    #: (左文本, x, y, w, h, 右文本, x, y, w, h)
    SAME_BUBBLE = [
        ("そんな平和な感じなら：", 173, 886, 61, 147, "いいんだけどな", 143, 889, 35, 166),
        ("片想いとか…", 504, 669, 40, 151, "幼馴染で", 536, 672, 35, 103),
        ("おま：は!?√元6", 672, 361, 77, 130, "ば…!", 745, 372, 33, 94),
        ("見てたしぁ…告白した!", 365, 377, 47, 130, "いつも", 408, 379, 26, 61),
        ("行ちぉか?", 1421, 610, 27, 147, "じゃああとでこそり", 1446, 612, 45, 108),
        ("こなくちゃ!", 1262, 637, 24, 97, "そう", 1284, 637, 18, 40),
        ("でももうすぐ消灯の時間なんじゃない?", 1441, 31, 80, 144, "うん", 1516, 34, 25, 72),
    ]

    def test_all_real_bubble_columns_merge(self):
        for t1, x1, y1, w1, h1, t2, x2, y2, w2, h2 in self.SAME_BUBBLE:
            with self.subTest(text=t1[:12]):
                merged = merge_vertical_columns([self.col(t1, x1, y1, w1, h1),
                                                 self.col(t2, x2, y2, w2, h2)])
                self.assertEqual(len(merged), 1, f"{t1} 与 {t2} 应合并成一句")

    def test_merged_text_reads_right_to_left(self):
        """右列在前。真实例子里「幼馴染で」在右、「片想いとか…」在左。"""
        merged = merge_vertical_columns([self.col("片想いとか…", 504, 669, 40, 151),
                                         self.col("幼馴染で", 536, 672, 35, 103)])
        self.assertEqual(merged[0]["text"], "幼馴染で片想いとか…")

    def test_wide_multi_column_box_is_not_treated_as_horizontal(self):
        """「おま」框宽 77 > 高 130÷2，旧规则误判成横排；框宽含 2~3 列是正常的。"""
        merged = merge_vertical_columns([self.col("おま：は!?√元6", 672, 361, 77, 130),
                                         self.col("ば…!", 745, 372, 33, 94)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "ば…!おま：は!?√元6")

    def test_glyph_size_matches_within_a_bubble(self):
        """字号估计在同气泡内一致、跨气泡差异明显——这是新判据的基础。"""
        from engine.ocr import _glyph_size
        same_a = _glyph_size(self.col("そんな平和な感じなら：", 173, 886, 61, 147))
        same_b = _glyph_size(self.col("いいんだけどな", 143, 889, 35, 166))
        self.assertLess(max(same_a, same_b) / min(same_a, same_b), 1.1)
        other = _glyph_size(self.col("短", 900, 900, 60, 120))       # 单字大字号
        self.assertGreater(max(same_a, other) / min(same_a, other), 1.5)


class MergeGuardTests(unittest.TestCase):
    """放宽合并规则后，这些情况仍必须保持独立。"""

    def col(self, text, x, y=80, w=44, h=335):
        return {"x": x, "y": y, "w": w, "h": h, "text": text,
                "score": 0.95, "vertical": True}

    def test_far_apart_bubbles_stay_separate(self):
        boxes = [self.col("おやすみなさい", 138), self.col("今日はとても楽", 552)]
        self.assertEqual(len(merge_vertical_columns(boxes)), 2)

    def test_vertically_stacked_bubbles_stay_separate(self):
        boxes = [self.col("上の台詞", 502, y=20, h=120), self.col("下の台詞", 552, y=300, h=120)]
        self.assertEqual(len(merge_vertical_columns(boxes)), 2)

    def test_staircase_alignment_stays_separate(self):
        boxes = [self.col("三の列", 100, y=100, h=200), self.col("二の列", 152, y=50, h=200),
                 self.col("一の列", 204, y=0, h=200)]
        self.assertEqual(len(merge_vertical_columns(boxes)), 3)

    def test_very_different_glyph_sizes_stay_separate(self):
        boxes = [self.col("窄列", 100, w=20), self.col("宽列", 130, w=44)]
        self.assertEqual(len(merge_vertical_columns(boxes)), 2)


if __name__ == "__main__":
    unittest.main()

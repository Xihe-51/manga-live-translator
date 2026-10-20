# -*- coding: utf-8 -*-
"""透明阅读 UI 的离屏测试；不抓取桌面，不调用翻译服务。"""
import base64
import copy
import ctypes
from ctypes import wintypes
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ui.reading import (
    NativePanel, ReadingOverlay, ReadingToolbar, WDA_EXCLUDEFROMCAPTURE,
    exclude_from_capture,
)

# discover 会与其他模块共用 QApplication，必须保存强引用且不能关闭应用。
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)
CFG = {"inpaint": {"pad": 2}, "capture": {"region": [-20, 30, 100, 60]},
       "llm": {"provider": "mock", "free": {}, "paid": {}, "mock": {}},
       # 本模块只测离屏合成结果（静态像素），关掉淡入让渲染即刻不透明；
       # 淡入动画本身由 test_regressions 的 MarkingOverlayTests 覆盖。
       "reading": {"fade_in": False}}


class FakeScreen:
    def __init__(self, origin=(0, 0), size=(800, 600), dpr=1.0):
        self._geometry = QRect(*origin, *size)
        self._dpr = dpr

    def geometry(self):
        return self._geometry

    def devicePixelRatio(self):
        return self._dpr


def encode_image(image):
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return "data:image/png;base64," + base64.b64encode(bytes(data)).decode()


def make_payload(boxes=None, color=QColor(200, 50, 25), original_color=None):
    image = QImage(100, 60, QImage.Format_RGBA8888)
    image.fill(color)
    painter = QPainter(image)
    painter.fillRect(QRect(44, 15, 3, 24), QColor(15, 80, 210))
    painter.end()
    payload = {"ok": True, "image": encode_image(image),
               "boxes": boxes if boxes is not None else [{"x": 20, "y": 15, "w": 10, "h": 15, "text": "原文"}],
               "translations": {"原文": "译文"}}
    if original_color is not None:
        original = QImage(100, 60, QImage.Format_RGBA8888)
        original.fill(original_color)
        payload["original"] = encode_image(original)
    return payload


def render_widget(widget):
    image = QImage(widget.size(), QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    widget.render(image)
    return image


class WidgetTestCase(unittest.TestCase):
    def keep(self, widget):
        self.addCleanup(widget.close)
        return widget


class OverlayTests(WidgetTestCase):
    def overlay(self, dpr=1.0):
        widget = self.keep(ReadingOverlay(CFG))
        widget.set_region([0, 0, 100, 60], FakeScreen(dpr=dpr))
        return widget

    def test_empty_background_is_transparent(self):
        widget = self.overlay()
        image = render_widget(widget)
        self.assertTrue(all(image.pixelColor(x, y).alpha() == 0
                            for y in range(image.height()) for x in range(image.width())))
        self.assertFalse(widget.autoFillBackground())

    def test_only_padded_text_patch_is_opaque(self):
        widget = self.overlay()
        widget.set_result(make_payload())
        image = render_widget(widget)
        for y in range(image.height()):
            for x in range(image.width()):
                expected = 255 if 18 <= x < 32 and 13 <= y < 32 else 0
                self.assertEqual(image.pixelColor(x, y).alpha(), expected, (x, y))
        self.assertEqual(image.pixelColor(22, 18), QColor(200, 50, 25))
        self.assertEqual(widget._patch.format(), QImage.Format_RGBA8888)

    def test_merged_box_keeps_chinese_in_column_gap_and_sources(self):
        sources = [{"x": 20, "y": 15, "w": 10, "h": 25},
                   {"x": 65, "y": 15, "w": 10, "h": 25},
                   {"x": 5, "y": 45, "w": 6, "h": 6}]
        merged = {"x": 20, "y": 15, "w": 55, "h": 25, "source_boxes": sources}
        widget = self.overlay()
        widget.set_result(make_payload([merged]))
        image = render_widget(widget)
        self.assertEqual(image.pixelColor(45, 25), QColor(15, 80, 210), "合并框内的中文不能被原始竖列 mask 截断")
        self.assertEqual(image.pixelColor(50, 20).alpha(), 255)
        self.assertEqual(image.pixelColor(6, 46).alpha(), 255, "不能遗漏合并框外的 source_boxes")
        self.assertEqual(image.pixelColor(90, 20).alpha(), 0)

    def test_top_level_source_boxes_and_clamping(self):
        widget = self.overlay()
        payload = make_payload([])
        payload["source_boxes"] = [{"x": -5, "y": -5, "w": 10, "h": 10},
                                   {"x": 98, "y": 58, "w": 10, "h": 10}]
        widget.set_result(payload)
        image = render_widget(widget)
        self.assertEqual(image.pixelColor(0, 0).alpha(), 255)
        self.assertEqual(image.pixelColor(6, 6).alpha(), 255)
        self.assertEqual(image.pixelColor(7, 7).alpha(), 0)
        self.assertEqual(image.pixelColor(99, 59).alpha(), 255)
        self.assertEqual(image.pixelColor(50, 30).alpha(), 0)

    def test_rgba_input_is_opaque_inside_patch(self):
        widget = self.overlay()
        widget.set_result(make_payload(color=QColor(200, 50, 25, 30)))
        self.assertEqual(render_widget(widget).pixelColor(22, 18).alpha(), 255)
        self.assertEqual(render_widget(widget).pixelColor(80, 50).alpha(), 0)

    def test_no_boxes_never_paints_full_screenshot(self):
        widget = self.overlay()
        widget.set_result(make_payload([]))
        image = render_widget(widget)
        self.assertTrue(all(image.pixelColor(x, y).alpha() == 0
                            for y in range(image.height()) for x in range(image.width())))

    def test_clear_and_new_region_remove_previous_patch(self):
        widget = self.overlay()
        widget.set_result(make_payload())
        widget.clear_result()
        self.assertTrue(widget._patch.isNull())
        self.assertEqual(render_widget(widget).pixelColor(22, 18).alpha(), 0)
        widget.set_result(make_payload())
        widget.set_region([20, 20, 100, 60], FakeScreen())
        self.assertTrue(widget._patch.isNull())
        self.assertEqual(render_widget(widget).pixelColor(22, 18).alpha(), 0)

    def test_result_replaces_instead_of_accumulating(self):
        widget = self.overlay()
        widget.set_result(make_payload())
        widget.set_result(make_payload([{"x": 75, "y": 45, "w": 10, "h": 10}]))
        image = render_widget(widget)
        self.assertEqual(image.pixelColor(22, 18).alpha(), 0)
        self.assertEqual(image.pixelColor(80, 50).alpha(), 255)

    def test_border_is_optional_and_survives_clear(self):
        widget = self.overlay()
        widget.set_border_visible(True)
        image = render_widget(widget)
        self.assertEqual(image.pixelColor(1, 20), QColor(55, 145, 255))
        self.assertEqual(image.pixelColor(50, 30).alpha(), 0)
        widget.set_result(make_payload())
        widget.clear_result()
        self.assertEqual(render_widget(widget).pixelColor(1, 20).alpha(), 255)
        widget.set_border_visible(False)
        self.assertEqual(render_widget(widget).pixelColor(1, 20).alpha(), 0)

    def test_hold_original_switches_between_patch_and_raw_frame(self):
        """按住对照键时整幅显示原图，松开立刻回到译文 patch。"""
        widget = self.overlay()
        widget.set_result(make_payload(original_color=QColor(10, 200, 60)))
        self.assertEqual(render_widget(widget).pixelColor(80, 50).alpha(), 0)
        widget.set_show_original(True)
        held = render_widget(widget)
        self.assertEqual(held.pixelColor(80, 50), QColor(10, 200, 60), "原图应铺满整个区域")
        self.assertEqual(held.pixelColor(22, 18), QColor(10, 200, 60))
        widget.set_show_original(False)
        self.assertEqual(render_widget(widget).pixelColor(22, 18), QColor(200, 50, 25))
        self.assertEqual(render_widget(widget).pixelColor(80, 50).alpha(), 0)

    def test_hold_original_falls_back_when_snapshot_missing(self):
        widget = self.overlay()
        widget.set_result(make_payload())            # 没有 original 字段
        widget.set_show_original(True)
        self.assertEqual(render_widget(widget).pixelColor(22, 18), QColor(200, 50, 25))
        payload = make_payload()
        payload["original"] = "%%%"
        widget.set_result(payload)                   # original 解析失败同样回退
        self.assertEqual(render_widget(widget).pixelColor(22, 18), QColor(200, 50, 25))

    def test_clear_result_drops_stale_original(self):
        widget = self.overlay()
        widget.set_result(make_payload(original_color=QColor(10, 200, 60)))
        widget.set_show_original(True)
        widget.clear_result()
        self.assertTrue(widget._original.isNull())
        self.assertEqual(render_widget(widget).pixelColor(22, 18).alpha(), 0)

    def test_window_flags_and_attributes(self):
        widget = self.overlay()
        self.assertEqual(widget.windowFlags() & Qt.WindowType_Mask, Qt.Tool)
        for flag in (Qt.FramelessWindowHint, Qt.WindowStaysOnTopHint,
                     Qt.WindowTransparentForInput, Qt.WindowDoesNotAcceptFocus):
            self.assertTrue(widget.windowFlags() & flag)
        for attribute in (Qt.WA_TranslucentBackground, Qt.WA_TransparentForMouseEvents,
                          Qt.WA_ShowWithoutActivating):
            self.assertTrue(widget.testAttribute(attribute))
        self.assertFalse(widget.capture_excluded)

    def test_dpr_mapping_and_negative_screen_origin(self):
        widget = self.keep(ReadingOverlay(CFG))
        screen = FakeScreen(origin=(-2400, 100), size=(1600, 1000), dpr=1.5)
        widget.set_region([-1800, 550, 1350, 825], screen)
        self.assertEqual(widget.geometry(), QRect(-2000, 400, 900, 550))
        self.assertEqual(widget._dpr, 1.5)
        positive = FakeScreen(origin=(2400, -200), size=(1600, 1000), dpr=2.0)
        widget.set_region([2700, 200, 600, 400], positive)
        self.assertEqual(widget.geometry(), QRect(2550, 0, 300, 200))

    def test_fractional_dpr_rounds_endpoints_consistently(self):
        widget = self.keep(ReadingOverlay(CFG))
        widget.set_region([-97, 54, 13, 17], FakeScreen(origin=(-100, 50), dpr=1.25))
        self.assertEqual(widget.geometry(), QRect(-98, 53, 11, 14))

    def test_automatic_screen_selection_and_cross_screen_rejection(self):
        widget = self.keep(ReadingOverlay(CFG))
        left = FakeScreen(origin=(-800, 0))
        right = FakeScreen()
        with patch("ui.reading.QGuiApplication.screens", return_value=[right, left]):
            widget.set_region([-600, 10, 100, 60])
            self.assertIs(widget._screen, left)
            with self.assertRaisesRegex(RuntimeError, "单屏"):
                widget.set_region([-30, 0, 100, 60])
        with self.assertRaisesRegex(RuntimeError, "跨屏"):
            widget.set_region([750, 0, 100, 60], right)
        self.assertEqual(widget.geometry(), QRect(-600, 10, 100, 60), "拒绝区域时不得修改原来的几何")
        widget.set_region([0, 0, 800, 600], right)
        self.assertEqual(widget.geometry(), QRect(0, 0, 800, 600))

    def test_invalid_regions(self):
        widget = self.overlay()
        for region in (None, [], [0, 0, 0, 60], [0, 0, -1, 60], [float("nan"), 0, 10, 10]):
            with self.subTest(region=region), self.assertRaises(ValueError):
                widget.set_region(region, FakeScreen())

    def test_patch_scales_with_physical_region(self):
        widget = self.overlay(dpr=2.0)
        widget.set_result(make_payload())
        image = render_widget(widget)
        self.assertEqual(widget.size().width(), 50)
        self.assertEqual(image.pixelColor(11, 9), QColor(200, 50, 25))
        self.assertEqual(image.pixelColor(40, 20).alpha(), 0)

    def test_invalid_image_or_payload_never_raises_and_clears(self):
        widget = self.overlay()
        for value in (None, 123, "", "%%%", "data:image/png;base64,%%%", "data:image/png,abc",
                      "data:image/png;base64,YWJj", b"not a png", "http://invalid/image.png"):
            with self.subTest(value=value):
                widget.set_result(make_payload())
                widget.set_result({"image": value, "boxes": make_payload()["boxes"]})
                self.assertTrue(widget._patch.isNull())
                self.assertEqual(render_widget(widget).pixelColor(22, 18).alpha(), 0)
        for payload in (None, [], {"ok": False, "error": "失败"}, {}):
            widget.set_result(payload)
            self.assertTrue(widget._patch.isNull())

    def test_malformed_boxes_are_ignored(self):
        widget = self.overlay()
        widget.set_result(make_payload([None, {}, {"x": "bad"}, {"x": 0, "y": 0, "w": -1, "h": 1},
                                        {"x": float("inf"), "y": 0, "w": 1, "h": 1}]))
        self.assertEqual(render_widget(widget).pixelColor(0, 0).alpha(), 0)

    def test_capture_affinity_after_first_native_handle(self):
        widget = self.keep(ReadingOverlay(CFG))
        self.assertFalse(widget.capture_excluded)
        with patch("ui.reading.exclude_from_capture", return_value=True) as exclude:
            widget.winId()
            self.assertTrue(widget.capture_excluded)
            exclude.assert_called_once_with(widget)
            widget.winId()
            self.assertEqual(exclude.call_count, 1)


class CaptureExclusionTests(unittest.TestCase):
    def api(self, set_ok=True, get_ok=True, actual=WDA_EXCLUDEFROMCAPTURE):
        def verify(hwnd, out):
            ctypes.cast(out, ctypes.POINTER(wintypes.DWORD)).contents.value = actual
            return get_ok
        return SimpleNamespace(SetWindowDisplayAffinity=Mock(return_value=set_ok),
                               GetWindowDisplayAffinity=Mock(side_effect=verify))

    def call_api(self, api, handle=0x1234567887654321):
        widget = SimpleNamespace(winId=lambda: handle)
        with patch("ui.reading.sys.platform", "win32"), \
                patch("ui.reading.QGuiApplication.platformName", return_value="windows"), \
                patch("ui.reading.ctypes.WinDLL", return_value=api, create=True) as load:
            result = exclude_from_capture(widget)
            load.assert_called_once_with("user32", use_last_error=True)
        return result

    def test_verified_affinity_uses_pointer_sized_hwnd(self):
        api = self.api()
        self.assertTrue(self.call_api(api))
        self.assertEqual(api.SetWindowDisplayAffinity.argtypes, [wintypes.HWND, wintypes.DWORD])
        self.assertEqual(api.SetWindowDisplayAffinity.restype, wintypes.BOOL)
        self.assertEqual(api.GetWindowDisplayAffinity.argtypes,
                         [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)])
        hwnd, affinity = api.SetWindowDisplayAffinity.call_args.args
        self.assertIsInstance(hwnd, wintypes.HWND)
        self.assertEqual(hwnd.value, 0x1234567887654321)
        self.assertEqual(affinity, 0x11)

    def test_failures_and_affinity_mismatch_return_false(self):
        for set_ok, get_ok, actual in ((False, True, 0x11), (True, False, 0x11), (True, True, 0x1)):
            with self.subTest(set_ok=set_ok, get_ok=get_ok, actual=actual):
                api = self.api(set_ok, get_ok, actual)
                self.assertFalse(self.call_api(api))
                if not set_ok:
                    api.GetWindowDisplayAffinity.assert_not_called()

    def test_offscreen_and_non_windows_do_not_load_windows_api(self):
        widget = SimpleNamespace(winId=Mock(return_value=1))
        for platform, qt_platform in (("win32", "offscreen"), ("win32", "minimal"), ("linux", "windows")):
            with self.subTest(platform=platform, qt_platform=qt_platform), \
                    patch("ui.reading.sys.platform", platform), \
                    patch("ui.reading.QGuiApplication.platformName", return_value=qt_platform), \
                    patch("ui.reading.ctypes.WinDLL", create=True) as load:
                self.assertFalse(exclude_from_capture(widget))
                load.assert_not_called()
        widget.winId.assert_not_called()

    def test_os_errors_and_invalid_handle_return_false(self):
        with patch("ui.reading.sys.platform", "win32"), \
                patch("ui.reading.QGuiApplication.platformName", return_value="windows"), \
                patch("ui.reading.ctypes.WinDLL", side_effect=OSError("不可用"), create=True) as load:
            self.assertFalse(exclude_from_capture(SimpleNamespace(winId=lambda: 1)))
            self.assertFalse(exclude_from_capture(SimpleNamespace(winId=lambda: 0)))
            self.assertEqual(load.call_count, 1)


class ToolbarTests(WidgetTestCase):
    def toolbar(self):
        return self.keep(ReadingToolbar())

    def test_independent_non_passthrough_toolbar(self):
        widget = self.toolbar()
        self.assertIsNone(widget.parentWidget())
        self.assertEqual(widget.width(), ReadingToolbar.DEFAULT_WIDTH)
        self.assertLessEqual(widget.height(), 80, "工具栏应为小条，不能保留 QWidget 默认高度")
        self.assertEqual(widget.windowFlags() & Qt.WindowType_Mask, Qt.Tool)
        self.assertTrue(widget.windowFlags() & Qt.WindowStaysOnTopHint)
        self.assertTrue(widget.windowFlags() & Qt.WindowDoesNotAcceptFocus)
        self.assertTrue(widget.testAttribute(Qt.WA_ShowWithoutActivating))
        self.assertFalse(widget.windowFlags() & Qt.WindowTransparentForInput)
        self.assertFalse(widget.testAttribute(Qt.WA_TransparentForMouseEvents))
        self.assertFalse(widget.testAttribute(Qt.WA_TranslucentBackground))
        widget.resize(ReadingToolbar.DEFAULT_WIDTH, widget.sizeHint().height())
        self.assertEqual(render_widget(widget).pixelColor(0, 0).alpha(), 255)

    def test_all_button_and_checkbox_signals(self):
        widget = self.toolbar()
        widget.set_ready(True)
        for button, signal in ((widget.toggle_button, widget.toggle_requested),
                               (widget.region_button, widget.region_requested),
                               (widget.refresh_button, widget.refresh_requested),
                               (widget.settings_button, widget.settings_requested),
                               (widget.exit_button, widget.exit_requested)):
            received = []
            signal.connect(lambda received=received: received.append(True))
            QTest.mouseClick(button, Qt.LeftButton)
            self.assertEqual(received, [True], button.text())
        borders, hidden = [], []
        widget.border_changed.connect(borders.append)
        widget.hide_text_changed.connect(hidden.append)
        widget.border_checkbox.setChecked(True)
        widget.border_checkbox.setChecked(False)
        widget.hide_text_checkbox.setChecked(True)
        self.assertEqual(borders, [True, False])
        self.assertEqual(hidden, [True])

    def test_original_hold_button_emits_press_and_release(self):
        """按住型按钮：按下 True、松开 False，与按住对照键行为一致。"""
        widget = self.toolbar()
        events = []
        widget.original_hold.connect(events.append)
        QTest.mousePress(widget.original_button, Qt.LeftButton)
        self.assertEqual(events, [True])
        QTest.mouseRelease(widget.original_button, Qt.LeftButton)
        self.assertEqual(events, [True, False])

    def test_original_hold_tooltip_shows_configured_key(self):
        widget = self.keep(ReadingToolbar({"reading": {"original_key": "ctrl+tab"}}))
        self.assertIn("Ctrl+Tab", widget.original_button.toolTip())
        default = self.toolbar()
        self.assertIn("未配置", default.original_button.toolTip())

    def test_ready_running_status_and_initial_configuration(self):
        widget = self.keep(ReadingToolbar({"reading": {"border_visible": True, "hide_text": True}}))
        self.assertTrue(widget.border_checkbox.isChecked())
        self.assertTrue(widget.hide_text_checkbox.isChecked())
        self.assertFalse(widget.toggle_button.isEnabled())
        widget.set_ready(True)
        self.assertTrue(widget.toggle_button.isEnabled())
        widget.set_running(True)
        self.assertEqual(widget.toggle_button.text(), "暂停")
        widget.set_ready(False)
        self.assertTrue(widget.toggle_button.isEnabled(), "运行中即使未ready也必须能暂停")
        self.assertFalse(widget.refresh_button.isEnabled())
        widget.set_running(False)
        self.assertEqual(widget.toggle_button.text(), "开始")
        self.assertFalse(widget.toggle_button.isEnabled())
        widget.set_status("等待翻页")
        self.assertEqual(widget.status_label.toolTip(), "等待翻页")

    def test_toolbar_capture_exclusion_on_handle_creation(self):
        widget = self.toolbar()
        self.assertFalse(widget.capture_excluded)
        with patch("ui.reading.exclude_from_capture", return_value=True) as exclude:
            widget.winId()
            self.assertTrue(widget.capture_excluded)
            exclude.assert_called_once_with(widget)

    def test_drag_label_moves_toolbar_without_emitting_actions(self):
        widget = self.toolbar()
        widget.move(100, 100)
        got = []
        widget.toggle_requested.connect(lambda: got.append(True))
        label = widget.drag_label
        def mouse(kind, global_point, button, buttons):
            return QMouseEvent(kind, QPointF(5, 5), QPointF(*global_point), button, buttons, Qt.NoModifier)
        from PySide6.QtCore import QEvent
        label.mousePressEvent(mouse(QEvent.MouseButtonPress, (110, 110), Qt.LeftButton, Qt.LeftButton))
        label.mouseMoveEvent(mouse(QEvent.MouseMove, (150, 170), Qt.NoButton, Qt.LeftButton))
        self.assertEqual(widget.pos(), QPoint(140, 160))
        label.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, (150, 170), Qt.LeftButton, Qt.NoButton))
        label.mouseMoveEvent(mouse(QEvent.MouseMove, (200, 200), Qt.NoButton, Qt.NoButton))
        self.assertEqual(widget.pos(), QPoint(140, 160))
        self.assertEqual(got, [])


class NativePanelTests(WidgetTestCase):
    def panel(self):
        return self.keep(NativePanel(CFG))

    def test_fields_provider_and_request_signals(self):
        widget = self.panel()
        self.assertEqual([field.text() for field in widget.fields], ["-20", "30", "100", "60"])
        self.assertEqual(widget.provider_combo.currentData(), "mock")
        widget.set_region("[-80，50，200，120]")
        widget.set_ready(True)
        picked, singles, realtime, providers, exited = [], [], [], [], []
        widget.pick_requested.connect(lambda: picked.append(True))
        widget.single_requested.connect(singles.append)
        widget.realtime_requested.connect(realtime.append)
        widget.provider_requested.connect(providers.append)
        widget.exit_requested.connect(lambda: exited.append(True))
        widget.pick_button.click()
        widget.single_button.click()
        widget.realtime_button.click()
        widget.provider_combo.setCurrentIndex(widget.provider_combo.findData("paid"))
        widget.exit_button.click()
        self.assertEqual(picked, [True])
        self.assertEqual(singles, ["-80,50,200,120"])
        self.assertEqual(realtime, ["-80,50,200,120"])
        self.assertEqual(providers, ["paid"])
        self.assertEqual(exited, [True])

    def test_inpaint_combo_reflects_config_and_emits(self):
        """抹字方式要能在设置页直接切，不用手改 config.yaml。"""
        widget = self.panel()
        self.assertEqual(widget.inpaint_combo.currentData(), "cover")
        methods = []
        widget.inpaint_requested.connect(methods.append)
        widget.inpaint_combo.setCurrentIndex(widget.inpaint_combo.findData("lama"))
        self.assertEqual(methods, ["lama"])

    def test_language_and_grain_controls_emit_and_lock(self):
        widget = self.panel()
        self.assertEqual(widget.source_language_combo.currentData(), "auto")
        self.assertFalse(widget.grain_check.isChecked())
        languages, grains = [], []
        widget.source_language_requested.connect(languages.append)
        widget.grain_requested.connect(grains.append)
        widget.source_language_combo.setCurrentIndex(widget.source_language_combo.findData("ja"))
        widget.grain_check.setChecked(True)
        self.assertEqual(languages, ["ja"])
        self.assertEqual(grains, [True])
        widget.set_busy(True)
        self.assertFalse(widget.source_language_combo.isEnabled())
        self.assertFalse(widget.grain_check.isEnabled())
        widget.set_busy(False)
        widget.set_realtime(True)
        self.assertFalse(widget.source_language_combo.isEnabled())
        self.assertFalse(widget.grain_check.isEnabled())

    def test_inpaint_combo_keeps_unknown_method_from_config(self):
        cfg = copy.deepcopy(CFG)
        cfg["inpaint"]["method"] = "cv2"
        widget = self.keep(NativePanel(cfg))
        self.assertEqual(widget.inpaint_combo.currentData(), "cv2")

    def test_busy_disables_mutation_but_realtime_can_pause(self):
        widget = self.panel()
        self.assertFalse(widget.single_button.isEnabled())
        self.assertFalse(widget.realtime_button.isEnabled())
        widget.set_ready(True)
        widget.set_busy(True)
        for control in (*widget.fields, widget.pick_button, widget.single_button, widget.provider_combo,
                        widget.inpaint_combo, widget.realtime_button):
            self.assertFalse(control.isEnabled())
        widget.set_realtime(True)
        widget.set_ready(False)
        self.assertTrue(widget.realtime_button.isEnabled())
        self.assertEqual(widget.realtime_button.text(), "暂停实时")
        requested = []
        widget.realtime_requested.connect(requested.append)
        widget.realtime_button.click()
        self.assertEqual(requested, ["-20,30,100,60"])
        self.assertTrue(widget._realtime, "点击不能自行切换状态，应由主调度确认")
        widget.set_realtime(False)
        widget.set_busy(False)
        widget.set_ready(True)
        self.assertTrue(all(control.isEnabled() for control in (*widget.fields, widget.pick_button,
                                                                widget.single_button, widget.provider_combo,
                                                                widget.inpaint_combo)))

    def test_result_image_translations_and_pairs_preview(self):
        widget = self.panel()
        widget.show_result(json.dumps(make_payload(), ensure_ascii=False))
        self.assertFalse(widget.preview_label.pixmap().isNull())
        self.assertEqual(widget.preview_label.pixmap().size().width(), 100)
        self.assertIn("原文\n译文", widget.pairs_view.toPlainText())
        payload = make_payload()
        payload["pairs"] = [{"source": "日文", "translation": "中文"}, {"src": "第二句", "dst": "译文二"},
                            ["第三句", "译文三"]]
        widget.show_result(json.dumps(payload, ensure_ascii=False))
        text = widget.pairs_view.toPlainText()
        self.assertIn("日文\n中文", text)
        self.assertIn("第二句\n译文二", text)
        self.assertIn("第三句\n译文三", text)

    def test_invalid_result_clears_old_preview_and_logs_plain_text(self):
        widget = self.panel()
        widget.show_result(json.dumps(make_payload()))
        widget.show_result('{"image":"%%%","pairs":{}}')
        self.assertTrue(widget.preview_label.pixmap().isNull())
        self.assertEqual(widget.pairs_view.toPlainText(), "")
        for result in ("bad JSON", "[]", "null", None, '{"ok":false,"error":"失败"}'):
            widget.show_result(result)
        widget.append_log("<b>只作为日志文字</b>")
        self.assertIn("结果读取失败", widget.log_view.toPlainText())
        self.assertIn("出错：失败", widget.log_view.toPlainText())
        self.assertIn("<b>只作为日志文字</b>", widget.log_view.toPlainText())

    def test_connect_button_and_presets_combo(self):
        """连接/模型按钮与预设下拉：预设一键填充并同步通道，重复选择可用。"""
        cfg = copy.deepcopy(CFG)
        cfg["llm"]["presets"] = [
            {"name": "DeepSeek", "provider": "paid", "base_url": "https://api.deepseek.com/v1",
             "model": "deepseek-chat", "options": {"temperature": 0}},
            {"name": "DeepSeek", "provider": "paid", "model": "重复名忽略"},
            {"name": "", "provider": "free"},
            "不是字典",
        ]
        widget = self.keep(NativePanel(cfg))
        self.assertEqual(widget.presets_combo.count(), 2, "占位项 + 一个有效预设，重复/无名/非字典项忽略")
        self.assertIsNone(widget.presets_combo.currentData())
        connects, presets, providers = [], [], []
        widget.test_connection_requested.connect(lambda: connects.append(True))
        widget.preset_requested.connect(presets.append)
        widget.provider_requested.connect(providers.append)
        widget.connect_button.click()
        self.assertEqual(connects, [True])
        widget._apply_preset(0)
        self.assertEqual(presets, [], "占位项不触发预设")
        widget._apply_preset(1)
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["name"], "DeepSeek")
        self.assertIsNone(widget.presets_combo.currentData(), "选完复位，方便重复选择")
        self.assertEqual(widget.provider_combo.currentData(), "paid", "通道下拉随预设同步")
        self.assertEqual(providers, ["paid"])

    def test_connect_and_presets_lock_with_busy(self):
        widget = self.panel()
        widget.set_busy(True)
        self.assertFalse(widget.presets_combo.isEnabled())
        self.assertFalse(widget.connect_button.isEnabled())
        widget.set_busy(False)
        self.assertTrue(widget.presets_combo.isEnabled())
        self.assertTrue(widget.connect_button.isEnabled())

    def test_show_models_result_branches(self):
        widget = self.panel()
        chosen = []
        widget.model_requested.connect(chosen.append)
        with patch("ui.reading.QMessageBox.warning") as warning:
            widget.show_models_result("bad JSON")
            widget.show_models_result("[]")
            widget.show_models_result('{"ok": false, "error": "鉴权失败，请检查该通道的 api_key"}')
        self.assertEqual(warning.call_count, 3)
        self.assertIn("连接测试失败：结果格式错误", widget.log_view.toPlainText())
        self.assertIn("连接测试失败：鉴权失败", widget.log_view.toPlainText())
        with patch("ui.reading.QMessageBox.information") as info:
            widget.show_models_result('{"ok": true, "provider": "paid", "models": []}')
        info.assert_called_once()
        with patch("ui.reading.QInputDialog.getItem",
                   return_value=("deepseek-reasoner", True)) as get_item:
            widget.show_models_result(json.dumps(
                {"ok": True, "provider": "paid", "current": "deepseek-reasoner",
                 "models": ["deepseek-chat", "deepseek-reasoner"]}))
        self.assertEqual(get_item.call_args[0][3], ["deepseek-chat", "deepseek-reasoner"])
        self.assertEqual(get_item.call_args[0][4], 1, "当前模型作为默认选中项")
        self.assertEqual(chosen, ["deepseek-reasoner"])
        with patch("ui.reading.QInputDialog.getItem", return_value=("deepseek-chat", False)):
            widget.show_models_result('{"ok": true, "provider": "paid", "models": ["deepseek-chat"]}')
        self.assertEqual(chosen, ["deepseek-reasoner"], "取消选择不发信号")


if __name__ == "__main__":
    unittest.main()

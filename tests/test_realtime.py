# -*- coding: utf-8 -*-
"""离屏实时回归：仅注入内存帧，不使用桌面抓屏、OCR 或网络。"""
import base64
import copy
import io
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QRect, QThread, Signal
from PySide6.QtTest import QTest
from PySide6.QtWebChannel import QWebChannel, QWebChannelAbstractTransport
from PySide6.QtWidgets import QApplication, QWidget

from engine.realtime import FrameTracker, frame_key
from ui.bridge import Bridge
from ui.realtime import RealtimeController

APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def frame(value, shape=(24, 32, 3)):
    return np.full(shape, value, dtype=np.uint8)


def image_value(payload):
    raw = base64.b64decode(payload["image"].split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        return image.getpixel((0, 0))[0]


class FakePipeline:
    def __init__(self):
        self.cfg = {}
        self.translator = SimpleNamespace(cfg={})
        self.calls = []
        self.configs = []
        self.gates = {}
        self.failures = set()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def block(self, index):
        entered, release = threading.Event(), threading.Event()
        self.gates[index] = (entered, release)
        return entered, release

    def release_all(self):
        for _, release in self.gates.values():
            release.set()

    def process(self, bgr, progress=None):
        with self.lock:
            index = len(self.calls)
            self.calls.append(bgr.copy())
            self.configs.append(copy.deepcopy(self.cfg))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if progress:
                progress("假流水线正在处理内存帧")
            if index in self.gates:
                entered, release = self.gates[index]
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("测试事件未释放")
            if index in self.failures:
                raise RuntimeError("模拟翻译失败")
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), [], []
        finally:
            with self.lock:
                self.active -= 1


class FakeBridge(Bridge):
    def __init__(self, cfg, panel):
        super().__init__(cfg, panel=panel)
        self.logs = []

    def _log(self, message):
        self.logs.append(message)
        self.log_line.emit(message)


class FakePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.ready = None
        self.realtime = None

    def set_ready(self, ready):
        self.ready = ready

    def set_realtime(self, active):
        self.realtime = active


class FakeCaptureWorker(QThread):
    sampled = Signal(object)
    failed = Signal(str)

    def __init__(self, region, token, parent, bgr, release):
        super().__init__(parent)
        self.region = list(region)
        self.token = token
        self.bgr = bgr.copy()
        self.release = release
        self.entered = threading.Event()

    def run(self):
        self.entered.set()
        if self.release.wait(10):
            self.sampled.emit((self.token, self.bgr))
        else:
            self.failed.emit("测试截图事件未释放")


class RecordingTransport(QWebChannelAbstractTransport):
    def __init__(self):
        super().__init__()
        self.messages = []

    def sendMessage(self, message):
        self.messages.append(dict(message))


def connect_channel(bridge):
    """订阅真实 QWebChannel 的结果信号，同时校验信号索引可用。"""
    channel = QWebChannel()
    transport = RecordingTransport()
    channel.registerObject("bridge", bridge)
    channel.connectTo(transport)
    transport.messageReceived.emit({"type": 3, "id": 1}, transport)
    initialization = next(message for message in transport.messages
                          if message.get("type") == 10 and message.get("id") == 1)
    signals = initialization["data"]["bridge"]["signals"]
    signal_index = next(index for name, index in signals
                        if name.split("(", 1)[0] == "result_ready")
    transport.messageReceived.emit({"type": 7, "object": "bridge",
                                   "signal": signal_index}, transport)
    return channel, transport, signal_index


class FrameTrackerTests(unittest.TestCase):
    def test_change_increments_revision_before_stability(self):
        tracker = FrameTracker(stable_frames=3)
        self.assertEqual(tracker.observe(frame(0)), (True, False, 1))
        self.assertEqual(tracker.observe(frame(0)), (False, False, 1))
        self.assertEqual(tracker.observe(frame(0)), (False, True, 1))
        self.assertEqual(tracker.observe(frame(80)), (True, False, 2))
        self.assertEqual(tracker.stable_frames, 1)
        self.assertEqual(tracker.observe(frame(160)), (True, False, 3))
        self.assertEqual(tracker.observe(frame(160)), (False, False, 3))
        self.assertEqual(tracker.observe(frame(160)), (False, True, 3))

    def test_stability_requires_at_least_two_frames(self):
        tracker = FrameTracker(stable_frames=1)
        self.assertEqual(tracker.observe(frame(0)), (True, False, 1))
        self.assertEqual(tracker.observe(frame(0)), (False, True, 1))

    def test_slow_changes_accumulate_against_anchor(self):
        tracker = FrameTracker(pixel_threshold=20)
        tracker.observe(frame(0))
        for value in (8, 16, 19):
            self.assertEqual(tracker.observe(frame(value)), (False, True, 1))
            np.testing.assert_array_equal(tracker.anchor, np.zeros((24, 32), np.uint8))
        self.assertEqual(tracker.observe(frame(24)), (True, False, 2))
        self.assertEqual(tracker.observe(frame(32)), (False, True, 2))
        self.assertEqual(tracker.observe(frame(48)), (True, False, 3))

    def test_change_ratio_and_pixel_threshold_boundaries(self):
        tracker = FrameTracker(pixel_threshold=20, change_ratio=0.03)
        original = frame(0, (10, 10, 3))
        tracker.observe(original)
        changed = original.copy()
        changed[0, :3] = 19
        self.assertEqual(tracker.observe(changed), (False, True, 1))
        changed[0, :2] = 20
        self.assertEqual(tracker.observe(changed), (False, True, 1))
        changed[0, 2] = 20
        self.assertEqual(tracker.observe(changed), (True, False, 2))

    def test_shape_change_reset_and_input_ownership(self):
        tracker = FrameTracker()
        original = frame(0)
        tracker.observe(original)
        original[:] = 80
        self.assertEqual(tracker.observe(original), (True, False, 2))
        self.assertEqual(tracker.observe(frame(80, (32, 24, 3))), (True, False, 3))
        tracker.reset()
        self.assertIsNone(tracker.anchor)
        self.assertEqual((tracker.revision, tracker.stable_frames), (0, 0))
        self.assertEqual(tracker.observe(original), (True, False, 1))

    def test_invalid_or_empty_frames_do_not_change_state(self):
        tracker = FrameTracker()
        tracker.observe(frame(0))
        anchor = tracker.anchor.copy()
        invalid = (None, [], np.zeros((0, 32, 3), np.uint8),
                   np.zeros((24, 0, 3), np.uint8), np.zeros((24, 32), np.uint8),
                   np.zeros((24, 32, 4), np.uint8), np.zeros((24, 32, 1), np.uint8))
        for value in invalid:
            with self.subTest(shape=getattr(value, "shape", None)):
                with self.assertRaisesRegex(ValueError, "空画面"):
                    tracker.observe(value)
                self.assertEqual((tracker.revision, tracker.stable_frames), (1, 1))
                np.testing.assert_array_equal(tracker.anchor, anchor)


class FrameKeyTests(unittest.TestCase):
    def test_dimensions_are_part_of_key_even_with_identical_bytes(self):
        original = frame(0, (2, 6, 3))
        reshaped = original.reshape(3, 4, 3)
        self.assertEqual(original.tobytes(), reshaped.tobytes())
        self.assertNotEqual(frame_key(original), frame_key(reshaped))

    def test_single_pixel_difference_is_not_an_approximate_cache_hit(self):
        original = frame(0)
        changed = original.copy()
        changed[0, 0, 0] = 1
        self.assertNotEqual(frame_key(original), frame_key(changed))
        self.assertEqual(frame_key(original), frame_key(original.copy()))

    def test_noncontiguous_array_has_same_key_as_its_copy(self):
        original = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        view = original[:, ::2]
        self.assertFalse(view.flags.c_contiguous)
        self.assertEqual(frame_key(view), frame_key(view.copy()))


class RealtimeControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"capture": {"interval_ms": 500},
                    "reading": {"stable_frames": 2, "pixel_threshold": 20,
                                "change_ratio": 0.003, "page_cache_size": 2}}
        self.panel = FakePanel()
        self.panel.show()
        self.bridge = FakeBridge(self.cfg, self.panel)
        self.pipeline = FakePipeline()
        self.bridge._pipeline = self.pipeline
        self.controller = RealtimeController(self.bridge, self.panel, self.cfg)
        self.results = []
        self.busy = []
        self.running = []
        self.bridge.result_ready.connect(lambda payload: self.results.append(json.loads(payload)))
        self.bridge.busy_changed.connect(self.busy.append)
        self.controller.running_changed.connect(self.running.append)
        self.capture_frame = frame(200)
        self.capture_gate = None
        self.capture_releases = []
        self.captures = []
        self.capture_visibility = []
        self.capture_patch = patch("ui.realtime.CaptureWorker", side_effect=self.make_capture)
        self.capture_patch.start()
        self.addCleanup(self.capture_patch.stop)
        self.region_patch = patch.object(self.controller.overlay, "set_region",
                                         side_effect=self.set_region)
        self.region_mock = self.region_patch.start()
        self.addCleanup(self.region_patch.stop)
        for window in (self.controller.overlay, self.controller.toolbar):
            window.capture_excluded = False
        result_patch = patch.object(self.controller.overlay, "set_result",
                                    wraps=self.controller.overlay.set_result)
        self.presented = result_patch.start()
        self.addCleanup(result_patch.stop)
        clear_patch = patch.object(self.controller.overlay, "clear_result",
                                   wraps=self.controller.overlay.clear_result)
        self.cleared = clear_patch.start()
        self.addCleanup(clear_patch.stop)
        running_patch = patch.object(self.controller.toolbar, "set_running",
                                     wraps=self.controller.toolbar.set_running)
        self.toolbar_running = running_patch.start()
        self.addCleanup(running_patch.stop)
        self.addCleanup(self.clean_up)

    def set_region(self, region):
        # 不让真实 set_region 依赖 offscreen 的 800x800 屏幕选区换算。
        self.controller.overlay._region = list(region)
        self.controller.overlay._screen = SimpleNamespace(
            availableGeometry=lambda: QRect(0, 0, 800, 800))
        self.controller.overlay.setGeometry(QRect(*region))

    def make_capture(self, region, token, parent):
        release = self.capture_gate or threading.Event()
        if self.capture_gate is None:
            release.set()
        self.capture_releases.append(release)
        worker = FakeCaptureWorker(region, token, parent, self.capture_frame, release)
        self.captures.append(worker)
        self.capture_visibility.append((self.controller.overlay.isVisible(),
                                        self.controller.toolbar.isVisible()))
        return worker

    def wait_for(self, predicate, timeout=4000):
        deadline = time.monotonic() + timeout / 1000
        while not predicate() and time.monotonic() < deadline:
            # QTest.qWait 在部分 PySide 构建中不释放 GIL，显式让 Python worker 获得执行机会。
            threading.Event().wait(0.001)
            QTest.qWait(10)
        self.assertTrue(predicate(), "等待 Qt 工作线程或信号超时")

    def start(self):
        # timer.stop 不能取消 start 中的 singleShot，必须同时隔离该自动采样入口。
        with patch("ui.realtime.QTimer.singleShot") as single_shot:
            self.controller.start("10,100,32,24")
            single_shot.assert_called_once()
        self.controller.timer.stop()
        self.controller._delay.stop()
        self.assertTrue(self.controller.running)

    def stable(self, value):
        bgr = frame(value) if isinstance(value, int) else value
        for _ in range(self.controller.tracker.stable_needed):
            self.controller.accept_frame(bgr)
        return bgr

    def finish_job(self):
        self.wait_for(lambda: self.controller._job is None and not self.bridge._busy)

    def clean_up(self):
        self.controller.pause()
        self.pipeline.release_all()
        for release in self.capture_releases:
            release.set()
        self.controller.shutdown()
        self.bridge.shutdown()
        self.wait_for(lambda: self.controller._job is None and self.controller._capture is None
                      and not self.bridge._busy)
        self.assertEqual(self.pipeline.active, 0)
        self.controller.overlay.close()
        self.controller.toolbar.close()
        self.panel.close()
        QTest.qWait(10)

    def test_style_change_updates_config_and_reasks_page(self):
        """改字号/颜色后必须落到 cfg 并触发重排，否则界面不会变。"""
        self.start()
        self.stable(200)
        self.finish_job()
        generation = self.controller._generation
        self.controller.toolbar.apply_style(1.4, "#ff0000")
        self.assertEqual(self.cfg["render"]["font_scale"], 1.4)
        self.assertEqual(self.cfg["render"]["color"], "#ff0000")
        self.assertEqual(self.controller._session_cfg["render"]["font_scale"], 1.4)
        self.assertGreater(self.controller._generation, generation, "必须触发重排")
        self.assertTrue(any("译文样式已更新" in message for message in self.bridge.logs))

    def test_style_change_while_paused_only_updates_config(self):
        """没在阅读时只记配置，不应凭空触发重排。"""
        generation = self.controller._generation
        self.controller.toolbar.apply_style(1.2, "#00ff00")
        self.assertEqual(self.cfg["render"]["font_scale"], 1.2)
        self.assertEqual(self.controller._generation, generation)

    def test_start_hides_settings_shows_toolbar_and_snapshots_config(self):
        self.bridge._panel_hidden = True
        self.start()
        self.region_mock.assert_called_once_with([10, 100, 32, 24])
        self.assertFalse(self.panel.isVisible())
        self.assertFalse(self.bridge._panel_hidden)
        self.assertTrue(self.controller.overlay.isVisible())
        self.assertTrue(self.controller.toolbar.isVisible())
        self.assertTrue(self.bridge._reading_active)
        self.assertTrue(self.panel.ready)
        self.assertTrue(self.panel.realtime)
        self.assertEqual(self.running, [True])
        self.toolbar_running.assert_called_with(True)
        self.assertIs(self.pipeline.cfg, self.controller._session_cfg)
        self.assertIs(self.pipeline.translator.cfg, self.controller._session_cfg)
        self.assertIsNot(self.pipeline.cfg, self.cfg)
        self.cfg["reading"]["stable_frames"] = 8
        self.assertEqual(self.pipeline.cfg["reading"]["stable_frames"], 2)
        self.assertFalse(self.controller.timer.isActive())
        self.assertFalse(self.controller._delay.isActive())
        self.assertEqual(self.captures, [])
        self.assertEqual(self.pipeline.calls, [])

    def test_pause_clears_display_and_stops_all_timers(self):
        self.start()
        self.stable(0)
        self.finish_job()
        self.assertIsNotNone(self.controller._last_payload)
        generation = self.controller._generation
        self.controller._retry.start()
        self.controller.pause()
        self.assertFalse(self.controller.running)
        self.assertEqual(self.controller._generation, generation + 1)
        self.assertIsNone(self.controller._pending)
        self.assertIsNone(self.controller._last_payload)
        self.assertIsNone(self.controller.tracker.anchor)
        self.assertFalse(self.controller.overlay.isVisible())
        self.assertTrue(self.controller.toolbar.isVisible())
        self.assertFalse(self.bridge._reading_active)
        self.assertFalse(self.panel.realtime)
        self.toolbar_running.assert_called_with(False)
        self.assertEqual(self.running, [True, False])
        for timer in (self.controller.timer, self.controller._delay, self.controller._retry):
            self.assertFalse(timer.isActive())
        self.cleared.assert_called()
        self.controller.accept_frame(frame(80))
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertEqual(self.controller.tracker.revision, 0)

    def test_pause_when_not_running_does_not_log_misleading_message(self):
        """设置/重选/退出都会无条件调 pause()，没在跑时不能刷“已暂停”误导排查。"""
        self.assertFalse(self.controller.running)
        self.controller.pause()
        self.assertFalse(any("已暂停" in message for message in self.bridge.logs))
        self.controller.show_settings()
        self.controller.exit_requested()
        self.assertFalse(any("已暂停" in message for message in self.bridge.logs))

    def test_pause_while_running_still_reports(self):
        """真在阅读时暂停，日志必须保留，否则用户不知道停了。"""
        self.start()
        self.controller.pause()
        self.assertTrue(any("已暂停透明实时" in message for message in self.bridge.logs))

    def test_toolbar_exit_returns_to_settings_without_quitting(self):
        """工具栏「退出」只回设置页：不能关掉设置页，也不能发 closed 让程序退出。"""
        closed = []
        self.controller.closed.connect(lambda: closed.append(True))
        self.start()
        self.controller.leave_reading()
        self.assertFalse(self.controller.running)
        self.assertTrue(self.panel.isVisible(), "应回到设置页")
        self.assertFalse(self.controller.toolbar.isVisible())
        self.assertFalse(self.controller.overlay.isVisible())
        self.assertEqual(closed, [], "回设置页不等于退出程序")
        self.assertTrue(any("已退出透明阅读，回到设置页" in m for m in self.bridge.logs))

    def test_toolbar_exit_while_paused_still_returns_to_settings(self):
        """没在阅读时点退出同样要回到设置页，只是不谎报“识别已停止”。"""
        self.controller.leave_reading()
        self.assertTrue(self.panel.isVisible())
        self.assertFalse(any("刚才的识别已停止" in m for m in self.bridge.logs))

    def test_panel_exit_still_closes_panel_and_signals_closed(self):
        """设置页的「退出」才真正收摊：关设置页并发 closed。"""
        closed = []
        self.controller.closed.connect(lambda: closed.append(True))
        self.controller.exit_requested()
        self.assertFalse(self.panel.isVisible())
        self.assertEqual(closed, [True])

    def test_change_immediately_clears_old_result_before_stable_n_frames(self):
        self.controller.tracker.stable_needed = 3
        self.start()
        self.controller.accept_frame(frame(0))
        self.controller.accept_frame(frame(0))
        self.assertIsNone(self.controller._job)
        self.controller.accept_frame(frame(0))
        self.finish_job()
        self.assertEqual(len(self.results), 1)
        self.controller.accept_frame(frame(80))
        self.assertEqual(self.controller.tracker.revision, 2)
        self.assertIsNone(self.controller._last_payload)
        self.assertIsNone(self.controller._displayed_revision)
        self.assertIsNone(self.controller._job)
        self.controller.accept_frame(frame(80))
        self.assertEqual(len(self.pipeline.calls), 1)
        self.controller.accept_frame(frame(80))
        self.finish_job()
        self.assertEqual([image_value(data) for data in self.results], [0, 80])

    def test_one_job_and_only_latest_pending_owns_frame_copy(self):
        entered, release = self.pipeline.block(0)
        self.start()
        self.stable(0)
        self.wait_for(entered.is_set)
        old_job = self.controller._job
        self.stable(80)
        latest = self.stable(160)
        for _ in range(5):
            self.controller.accept_frame(latest)
        self.assertIs(self.controller._job, old_job)
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertEqual(self.controller._pending[:3],
                         (self.controller._generation, 3, frame_key(latest)))
        latest[:] = 240
        np.testing.assert_array_equal(self.controller._pending[3], frame(160))
        release.set()
        self.finish_job()
        self.assertEqual([int(bgr[0, 0, 0]) for bgr in self.pipeline.calls], [0, 160])
        self.assertEqual(self.pipeline.max_active, 1)
        self.assertEqual([image_value(data) for data in self.results], [160])
        self.assertNotIn(frame_key(frame(0)), self.controller._cache)
        self.assertIsNone(self.controller._pending)
        self.assertEqual(self.busy, [True, False, True, False])
        self.stable(160)
        self.assertEqual(len(self.pipeline.calls), 2)
        self.assertEqual(len(self.results), 1)

    def test_new_unstable_page_cancels_older_pending_and_old_progress(self):
        entered, release = self.pipeline.block(0)
        self.start()
        self.stable(0)
        self.wait_for(entered.is_set)
        self.stable(80)
        self.assertIsNotNone(self.controller._pending)
        self.controller.accept_frame(frame(160))
        self.assertIsNone(self.controller._pending)
        logs = list(self.bridge.logs)
        self.controller._job_progress("过时进度")
        self.assertEqual(self.bridge.logs, logs)
        release.set()
        self.finish_job()
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertEqual(self.results, [])
        self.controller.accept_frame(frame(160))
        self.finish_job()
        self.assertEqual([image_value(data) for data in self.results], [160])

    def test_realtime_job_reports_live_frame_not_frozen_shot(self):
        """实时模式用的是刚抓到的帧，进度提示不能说成“框选时冻结的截图”。"""
        self.start()
        self.stable(200)
        self.assertIsNotNone(self.controller._job)
        self.assertEqual(self.controller._job._source, "实时截图")
        self.assertTrue(self.controller._job._include_original)

    def test_realtime_result_carries_original_snapshot_for_hold_compare(self):
        """实时结果要带原图帧，按住对照键时才有东西可显示。"""
        self.start()
        self.stable(200)
        self.finish_job()
        payload = self.results[-1]
        self.assertIn("original", payload)
        raw = base64.b64decode(payload["original"].split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual(image.getpixel((0, 0))[0], 200, "原图应与抓到的帧一致")

    def test_hold_key_and_button_switch_overlay_to_original(self):
        """按住对照键或工具栏按钮都要切换覆盖层绘制源，松开立刻恢复。"""
        # 注入恒为松开的假键盘，避免真轮询在断言期间插入额外切换。
        self.controller._hold._user32 = SimpleNamespace(GetAsyncKeyState=lambda vk: 0)
        self.start()
        with patch.object(self.controller.overlay, "set_show_original") as show:
            self.controller._hold.pressed.emit()
            self.controller._hold.released.emit()
            self.controller.toolbar.original_hold.emit(True)
            self.controller.toolbar.original_hold.emit(False)
        self.assertEqual([call.args[0] for call in show.call_args_list],
                         [True, False, True, False])
        self.assertTrue(any("可看原图" in message for message in self.bridge.logs))

    def test_hold_monitor_runs_only_while_reading(self):
        """轮询只在阅读期间跑：暂停必须停表，别一直啃键盘状态。"""
        self.controller._hold._user32 = SimpleNamespace(GetAsyncKeyState=lambda vk: 0)
        self.start()
        self.assertTrue(self.controller._hold.timer.isActive())
        self.controller.pause()
        self.assertFalse(self.controller._hold.timer.isActive())

    def test_pause_discards_inflight_result_and_pending(self):
        entered, release = self.pipeline.block(0)
        self.start()
        self.stable(0)
        self.wait_for(entered.is_set)
        self.stable(80)
        self.controller.pause()
        self.assertIsNone(self.controller._pending)
        release.set()
        self.finish_job()
        self.assertEqual(self.results, [])
        self.presented.assert_not_called()
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertEqual(dict(self.controller._cache), {})
        self.assertFalse(self.controller.overlay.isVisible())
        self.assertFalse(self.bridge._busy)

    def test_refresh_discards_old_generation_even_when_revision_matches(self):
        entered, release = self.pipeline.block(0)
        self.start()
        self.stable(0)
        self.wait_for(entered.is_set)
        token = self.controller._job_token
        self.controller.refresh()
        self.assertEqual(dict(self.controller._cache), {})
        self.assertIsNone(self.controller._last_payload)
        self.stable(80)
        self.assertEqual(self.controller.tracker.revision, token[1])
        self.assertNotEqual(self.controller._generation, token[0])
        release.set()
        self.finish_job()
        self.assertEqual([int(bgr[0, 0, 0]) for bgr in self.pipeline.calls], [0, 80])
        self.assertEqual([image_value(data) for data in self.results], [80])
        self.assertNotIn(frame_key(frame(0)), self.controller._cache)
        self.assertEqual(self.pipeline.max_active, 1)

    def test_page_cache_reuses_exact_page_and_evicts_least_recently_used(self):
        self.start()
        for value in (0, 80):
            self.stable(value)
            self.finish_job()
        self.stable(0)
        self.assertIsNone(self.controller._job)
        self.assertEqual(len(self.pipeline.calls), 2)
        self.assertEqual([image_value(data) for data in self.results], [0, 80, 0])
        self.assertEqual(list(self.controller._cache), [frame_key(frame(80)), frame_key(frame(0))])
        self.stable(160)
        self.finish_job()
        self.assertEqual(list(self.controller._cache), [frame_key(frame(0)), frame_key(frame(160))])
        self.stable(80)
        self.finish_job()
        self.assertEqual(len(self.pipeline.calls), 4)
        self.controller.refresh()
        self.assertEqual(dict(self.controller._cache), {})
        self.stable(80)
        self.finish_job()
        self.assertEqual(len(self.pipeline.calls), 5)
        self.controller.pause()
        self.start()
        self.assertEqual(dict(self.controller._cache), {})

    def test_page_cache_can_be_disabled(self):
        self.controller._cache_limit = 0
        self.start()
        for value in (0, 80, 0):
            self.stable(value)
            self.finish_job()
        self.assertEqual(len(self.pipeline.calls), 3)
        self.assertEqual(dict(self.controller._cache), {})

    def test_failure_waits_two_seconds_and_retries_only_latest_pending(self):
        self.pipeline.failures.add(0)
        self.start()
        self.stable(0)
        self.finish_job()
        self.assertEqual(len(self.results), 1)
        self.assertFalse(self.results[0]["ok"])
        self.assertIn("模拟翻译失败", self.results[0]["error"])
        self.assertTrue(self.controller._retry.isActive())
        self.assertEqual(self.controller._retry.interval(), 2000)
        retry_started = time.monotonic()
        self.stable(0)
        self.stable(80)
        self.stable(160)
        for _ in range(5):
            self.stable(160)
        QTest.qWait(150)
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertIsNone(self.controller._job)
        self.assertEqual(self.controller._pending[2], frame_key(frame(160)))
        self.wait_for(lambda: len(self.pipeline.calls) == 2, timeout=3000)
        self.assertGreaterEqual(time.monotonic() - retry_started, 1.7)
        self.finish_job()
        self.assertEqual([int(bgr[0, 0, 0]) for bgr in self.pipeline.calls], [0, 160])
        self.assertEqual(image_value(self.results[-1]), 160)
        self.assertEqual(len(self.results), 2)
        self.assertEqual(self.pipeline.max_active, 1)
        self.assertFalse(self.controller._retry.isActive())
        self.assertIsNone(self.controller._pending)

    def test_pause_cancels_failed_job_retry(self):
        self.pipeline.failures.add(0)
        self.start()
        self.stable(0)
        self.finish_job()
        self.stable(80)
        self.assertTrue(self.controller._retry.isActive())
        self.controller.pause()
        self.assertFalse(self.controller._retry.isActive())
        self.assertIsNone(self.controller._pending)
        self.controller._dispatch()
        self.assertEqual(len(self.pipeline.calls), 1)

    def test_old_capture_return_after_refresh_does_not_replace_new_page(self):
        self.capture_gate = threading.Event()
        self.start()
        self.controller._tick()
        self.wait_for(lambda: bool(self.captures) and self.captures[0].entered.is_set())
        old_generation = self.captures[0].token
        self.controller.refresh()
        self.stable(80)
        self.finish_job()
        revision = self.controller.tracker.revision
        clear_count = self.cleared.call_count
        self.assertNotEqual(old_generation, self.controller._generation)
        self.capture_gate.set()
        self.wait_for(lambda: self.controller._capture is None)
        self.assertEqual(self.controller.tracker.revision, revision)
        self.assertEqual(self.cleared.call_count, clear_count)
        self.assertEqual([image_value(data) for data in self.results], [80])
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertTrue(self.controller.overlay.isVisible())
        self.assertTrue(self.controller.toolbar.isVisible())

    def test_old_capture_return_after_pause_does_not_process_frame(self):
        self.capture_gate = threading.Event()
        self.start()
        self.controller._tick()
        self.wait_for(lambda: bool(self.captures) and self.captures[0].entered.is_set())
        self.controller.pause()
        self.capture_gate.set()
        self.wait_for(lambda: self.controller._capture is None)
        self.assertEqual(self.controller.tracker.revision, 0)
        self.assertEqual(self.pipeline.calls, [])
        self.assertEqual(self.results, [])
        self.assertFalse(self.controller.overlay.isVisible())
        self.assertTrue(self.controller.toolbar.isVisible())

    def test_fallback_hides_overlay_and_toolbar_until_fake_capture_returns(self):
        self.capture_gate = threading.Event()
        self.start()
        self.controller._tick()
        self.assertFalse(self.controller.overlay.isVisible())
        self.assertFalse(self.controller.toolbar.isVisible())
        self.assertEqual(self.controller._hidden_for_capture,
                         [self.controller.overlay, self.controller.toolbar])
        self.assertTrue(self.controller._delay.isActive())
        self.assertEqual(self.controller._delay.interval(), 75)
        self.assertIsNone(self.controller._capture)
        self.controller._tick()
        self.assertEqual(self.captures, [])
        self.wait_for(lambda: bool(self.captures) and self.captures[0].entered.is_set())
        self.assertEqual(self.capture_visibility, [(False, False)])
        self.assertEqual(self.captures[0].region, [10, 100, 32, 24])
        self.assertEqual(self.captures[0].token, self.controller._generation)
        self.controller._tick()
        self.assertEqual(len(self.captures), 1)
        self.capture_gate.set()
        self.wait_for(lambda: self.controller._capture is None)
        self.assertTrue(self.controller.overlay.isVisible())
        self.assertTrue(self.controller.toolbar.isVisible())
        self.assertEqual(self.controller._hidden_for_capture, [])
        self.assertEqual(self.controller.tracker.revision, 1)
        self.assertEqual(self.pipeline.calls, [])

    def test_capture_exclusion_hides_only_unsafe_window(self):
        self.capture_gate = threading.Event()
        self.start()
        # show() 会重新计算 offscreen 截图排除状态；开始后模拟平台确认成功。
        self.controller.overlay.capture_excluded = True
        self.controller._tick()
        self.assertTrue(self.controller.overlay.isVisible())
        self.assertFalse(self.controller.toolbar.isVisible())
        self.assertEqual(self.controller._hidden_for_capture, [self.controller.toolbar])
        self.wait_for(lambda: bool(self.captures) and self.captures[0].entered.is_set())
        self.assertEqual(self.capture_visibility, [(True, False)])
        self.capture_gate.set()
        self.wait_for(lambda: self.controller._capture is None)
        self.assertTrue(self.controller.toolbar.isVisible())

    def test_excluded_windows_capture_immediately_without_hiding(self):
        self.capture_gate = threading.Event()
        self.start()
        for window in (self.controller.overlay, self.controller.toolbar):
            window.capture_excluded = True
        self.controller._tick()
        self.assertFalse(self.controller._delay.isActive())
        self.assertEqual(self.controller._hidden_for_capture, [])
        self.assertEqual(self.capture_visibility, [(True, True)])
        self.capture_gate.set()
        self.wait_for(lambda: self.controller._capture is None)

    def test_pause_during_fallback_delay_restores_toolbar_without_capture(self):
        self.start()
        self.controller._tick()
        self.assertTrue(self.controller._delay.isActive())
        self.controller.pause()
        self.assertFalse(self.controller._delay.isActive())
        self.assertFalse(self.controller.overlay.isVisible())
        self.assertTrue(self.controller.toolbar.isVisible())
        self.controller._capture_now()
        QTest.qWait(100)
        self.assertEqual(self.captures, [])

    def test_capture_restore_respects_hide_text_checkbox(self):
        self.capture_gate = threading.Event()
        self.start()
        self.controller._tick()
        self.wait_for(lambda: bool(self.captures) and self.captures[0].entered.is_set())
        self.controller.toolbar.hide_text_checkbox.setChecked(True)
        self.capture_gate.set()
        self.wait_for(lambda: self.controller._capture is None)
        self.assertFalse(self.controller.overlay.isVisible())
        self.assertTrue(self.controller.toolbar.isVisible())

    def test_visible_settings_panel_suppresses_sampling(self):
        self.start()
        self.panel.show()
        self.controller._tick()
        self.assertEqual(self.captures, [])
        self.assertFalse(self.controller._delay.isActive())
        self.assertTrue(self.controller.overlay.isVisible())
        self.assertTrue(self.controller.toolbar.isVisible())

    def test_single_page_frozen_png_result_remains_webchannel_compatible(self):
        channel, transport, result_index = connect_channel(self.bridge)
        entered, release = self.pipeline.block(0)
        bgr = frame(80)
        ok, encoded = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        self.bridge._selected_region = [10, 100, 32, 24]
        self.bridge._selected_png = encoded.tobytes()
        self.bridge.translate_page("10,100,32,24")
        self.wait_for(entered.is_set)
        self.assertTrue(self.bridge._busy)
        self.assertTrue(self.panel.isVisible())
        self.assertEqual(self.captures, [])
        release.set()
        self.wait_for(lambda: not self.bridge._busy and bool(self.results))
        self.bridge.shutdown()
        self.assertFalse(self.bridge._worker.isRunning())
        self.assertEqual(self.busy, [True, False])
        self.assertEqual(len(self.results), 1)
        payload = self.results[0]
        self.assertEqual(set(payload), {"ok", "image", "boxes", "translations"})
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["boxes"], [])
        self.assertEqual(payload["translations"], [])
        self.assertEqual(image_value(payload), 80)
        np.testing.assert_array_equal(self.pipeline.calls[0], bgr)
        self.assertEqual(self.pipeline.configs[0], self.cfg)
        self.assertTrue(self.panel.isVisible())
        self.wait_for(lambda: any(message.get("type") == 1
                                 and message.get("signal") == result_index
                                 for message in transport.messages))
        packet = next(message for message in transport.messages
                      if message.get("type") == 1 and message.get("signal") == result_index)
        self.assertEqual(packet["object"], "bridge")
        self.assertEqual(json.loads(packet["args"][0]), payload)
        channel.disconnectFrom(transport)


if __name__ == "__main__":
    unittest.main()

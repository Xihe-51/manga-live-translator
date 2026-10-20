# -*- coding: utf-8 -*-
"""透明阅读控制器：后台抓屏、单任务推理、只保留最新帧、安全截图降级。"""
import copy
import json
from collections import OrderedDict

import cv2
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from engine.realtime import FrameTracker, frame_key
from engine.hotkey import HotkeyManager, HoldKeyMonitor, hotkeys_from_config
from engine.configio import update_section_values
from .bridge import TranslateWorker, parse_region
from .reading import ReadingOverlay, ReadingToolbar


class CaptureWorker(QThread):
    sampled = Signal(object)
    failed = Signal(str)

    def __init__(self, region, token, parent=None):
        super().__init__(parent)
        self.region, self.token = list(region), token

    def run(self):
        cap = None
        try:
            from engine.capture import ScreenCapturer
            cap = ScreenCapturer()
            self.sampled.emit((self.token, cap.grab(self.region)))
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if cap is not None:
                cap.close()


class RealtimeController(QObject):
    running_changed = Signal(bool)
    closed = Signal()

    #: 热键动作名 → 控制器方法名。放在类属性上，UI 列快捷键时也能直接读。
    HOTKEY_ACTIONS = {
        "toggle": "toggle",
        "reselect": "reselect",
        "refresh": "refresh",
        "settings": "show_settings",
        "leave": "leave_reading",
    }

    def __init__(self, bridge, panel, cfg, parent=None):
        super().__init__(parent)
        self.bridge, self.panel, self.cfg = bridge, panel, cfg
        self.overlay = ReadingOverlay(cfg)
        self.toolbar = ReadingToolbar(cfg)
        reading = cfg.get("reading", {})
        self.tracker = FrameTracker(reading.get("stable_frames", 2),
                                    reading.get("pixel_threshold", 20), reading.get("change_ratio", 0.003))
        self.timer = QTimer(self)
        self.timer.setInterval(max(150, int(cfg.get("capture", {}).get("interval_ms", 500))))
        self.timer.timeout.connect(self._tick)
        self._delay = QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.timeout.connect(self._capture_now)
        self.running = False
        self._generation = 0
        self._region = None
        self._capture = None
        self._capture_token = None
        self._hidden_for_capture = []
        self._job = None
        self._job_token = None
        self._job_key = None
        self._pending = None
        self._displayed_revision = None
        self._last_payload = None
        self._cache = OrderedDict()
        self._cache_limit = min(16, max(0, int(reading.get("page_cache_size", 3))))
        self._session_cfg = None
        self._resume_after_pick = False
        self._session_pairs = OrderedDict()
        self._marks = OrderedDict()
        self._retry = QTimer(self)
        self._retry.setSingleShot(True)
        self._retry.setInterval(2000)
        self._retry.timeout.connect(self._dispatch)
        self.toolbar.toggle_requested.connect(self.toggle)
        self.toolbar.region_requested.connect(self.reselect)
        self.toolbar.settings_requested.connect(self.show_settings)
        self.toolbar.refresh_requested.connect(self.refresh)
        self.toolbar.export_requested.connect(self.export_session)
        # 工具栏的「退出」只退出透明阅读、回到设置页；真正退出程序在设置页。
        self.toolbar.exit_requested.connect(self.leave_reading)
        self.toolbar.border_changed.connect(self.overlay.set_border_visible)
        self.toolbar.hide_text_changed.connect(self._hide_text)
        self.toolbar.style_changed.connect(self._style_changed)
        # 原图对照：按钮与按住键（默认 Tab）走同一条切换路径。
        self.toolbar.original_hold.connect(self._show_original)
        # 译错标记：工具栏开标记模式，覆盖层点中某句时回传原文。
        self.toolbar.mark_requested.connect(self._toggle_marking)
        self.toolbar.peek_requested.connect(self._toggle_peeking)
        self.overlay.sentence_clicked.connect(self._on_sentence_clicked)
        self._hold = HoldKeyMonitor(reading.get("original_key", "tab"), parent=self)
        self._hold.pressed.connect(lambda: self._show_original(True))
        self._hold.released.connect(lambda: self._show_original(False))
        bridge.log_line.connect(self._status)
        bridge.region_ready.connect(self._region_selected)
        bridge.selection_finished.connect(self._selection_finished)
        bridge.model_ready.connect(self._ready)
        bridge.reading_mode_changed.connect(self._mode_changed)
        bridge.busy_changed.connect(self._busy_changed)
        self._hotkeys = HotkeyManager(hotkeys_from_config(cfg), parent=self)
        self._hotkeys.triggered.connect(self._on_hotkey)
        self._hotkeys.message.connect(bridge._log)
        self._ready(bridge._pipeline is not None)

    # ---------- 全局快捷键 ----------
    @property
    def hotkeys(self):
        return self._hotkeys

    def start_hotkeys(self):
        """启动时装载全局快捷键；失败只提示，不阻断程序启动。"""
        self._hotkeys.start()

    @Slot(str)
    def _on_hotkey(self, action):
        """热键触发的动作都会先看一下当前是否处于可执行状态，再调用既有槽。"""
        method = self.HOTKEY_ACTIONS.get(action)
        if method is None:
            return
        if not self._hotkey_allowed(action):
            return
        getattr(self, method)()

    def _hotkey_allowed(self, action):
        """可执行性判断：模型没就绪、正忙、或在框选时，热键不该硬闯。"""
        if action in ("toggle", "reselect") and self.bridge._pipeline is None:
            self.bridge._log("模型还在加载，快捷键暂时不可用")
            return False
        if self.bridge._picking:
            self.bridge._log("正在框选中，快捷键已忽略")
            return False
        if action == "toggle" and self.bridge._busy:
            self.bridge._log("当前任务尚未结束，快捷键已忽略")
            return False
        return True

    @Slot(bool)
    def _ready(self, ready):
        self.toolbar.set_ready(ready and not self.bridge._busy and self._job is None)
        if hasattr(self.panel, "set_ready"):
            self.panel.set_ready(ready)

    @Slot(bool)
    def _busy_changed(self, busy):
        self._ready(self.bridge._pipeline is not None)

    @Slot(str)
    def _status(self, message):
        self.toolbar.set_status(message)

    def _mode_changed(self, active):
        if hasattr(self.panel, "set_realtime"):
            self.panel.set_realtime(active)

    @Slot()
    def toggle(self):
        if self.running:
            self.pause()
        else:
            region = self._region or self.bridge._selected_region
            self.start(",".join(map(str, region)) if region else "")

    @Slot(str)
    def toggle_region(self, region):
        if self.running:
            self.pause()
        else:
            self.start(region)

    @Slot(str)
    def start(self, region_text):
        region = parse_region(region_text)
        if region is None:
            self.bridge._log("请先框选漫画阅读区域，再进入透明实时模式")
            return
        if self.bridge._pipeline is None or self.bridge._busy or self.bridge._picking or self._job is not None:
            self.bridge._log("模型或上一次任务尚未结束，请稍后开始实时阅读")
            return
        try:
            self.overlay.set_region(region)
        except Exception as error:
            self.bridge._log(str(error))
            return
        self._region = list(region)
        self._session_cfg = copy.deepcopy(self.cfg)
        self.bridge._pipeline.cfg = self._session_cfg
        self.bridge._pipeline.translator.cfg = self._session_cfg
        self._generation += 1
        self.tracker.reset()
        self._cache.clear()
        self._session_pairs.clear()
        self._marks.clear()
        self.overlay.set_marks(())
        self._pending = None
        self._displayed_revision = None
        self._last_payload = None
        self.running = True
        self.bridge._reading_active = True
        self.bridge.reading_mode_changed.emit(True)
        self.panel.hide()
        # 面板是用户有意收起的，不能被单页桥的“恢复面板”机制自动打开。
        self.bridge._panel_hidden = False
        screen = self.overlay._screen.availableGeometry()
        self.toolbar.move(max(screen.x(), min(self.overlay.x(), screen.right() - self.toolbar.width())),
                          max(screen.y(), self.overlay.y() - self.toolbar.height() - 6))
        self.overlay.show()
        self.toolbar.show()
        self.toolbar.set_running(True)
        self._ready(True)
        self.running_changed.emit(True)
        safe = self.overlay.capture_excluded and self.toolbar.capture_excluded
        hold_hint = f"；按住 {self._hold.label} 可看原图" if self._hold.available else ""
        self.bridge._log("透明实时已开始：画面稳定后自动翻译；" +
                         ("窗口已设置截图排除" if safe else "截图排除不可用，采用短暂隐藏悬浮层抓屏") +
                         hold_hint)
        self._hold.start()
        self.timer.start()
        QTimer.singleShot(160, self._tick)

    @Slot()
    def pause(self):
        was_running = self.running
        self.running = False
        self._generation += 1
        self.timer.stop()
        self._delay.stop()
        self._retry.stop()
        self._hold.stop()
        self._pending = None
        self.tracker.reset()
        self._last_payload = None
        self.overlay.clear_result()
        self.overlay.hide()
        self._restore_capture_windows()
        self.bridge._reading_active = False
        self.bridge.reading_mode_changed.emit(False)
        self.toolbar.set_running(False)
        self._ready(self.bridge._pipeline is not None)
        self.running_changed.emit(False)
        # show_settings/reselect/exit 都会无条件调 pause()，本来就没在跑时
        # 再报“已暂停”会把日志刷成假象，干扰排查。
        if not was_running:
            return
        if self._job is not None:
            self.bridge._log("已暂停；正在运行的识别会自然结束，结果不会再显示")
        else:
            self.bridge._log("已暂停透明实时，点开始可继续")

    @Slot()
    def _tick(self):
        if not self.running or self._capture is not None or self._delay.isActive():
            return
        # 设置页被打开时暂停采样，避免抓到设置页而丢失已翻译内容。
        if self.panel.isVisible():
            return
        self._capture_token = self._generation
        self._hidden_for_capture = []
        for window in (self.overlay, self.toolbar):
            if window.isVisible() and not window.capture_excluded:
                window.hide()
                self._hidden_for_capture.append(window)
        if self._hidden_for_capture:
            self._delay.start(75)  # 让桌面合成器先移除覆盖窗口，不阻塞GUI。
        else:
            self._capture_now()

    @Slot()
    def _capture_now(self):
        if not self.running or self._capture_token != self._generation:
            self._restore_capture_windows()
            return
        self._capture = CaptureWorker(self._region, self._capture_token, self)
        self._capture.sampled.connect(self._sampled)
        self._capture.failed.connect(self._capture_failed)
        self._capture.finished.connect(self._capture_finished)
        self._capture.start()

    def _restore_capture_windows(self):
        hidden, self._hidden_for_capture = self._hidden_for_capture, []
        if self.running:
            for window in hidden:
                if window is self.overlay and self.toolbar.hide_text_checkbox.isChecked():
                    continue
                window.show()
        elif self.toolbar in hidden:
            self.toolbar.show()

    @Slot(object)
    def _sampled(self, result):
        self._restore_capture_windows()
        token, bgr = result
        if not self.running or token != self._generation:
            return
        self.accept_frame(bgr)

    def accept_frame(self, bgr):
        """仅在主线程使用；测试可注入帧，不触碰用户桌面。"""
        if not self.running:
            return
        changed, stable, revision = self.tracker.observe(bgr)
        if changed:
            self.overlay.clear_result()
            self._last_payload = None
            self._pending = None
            self._displayed_revision = None
            self.toolbar.set_status("画面变化，等待滚动停止…")
        if not stable or self._displayed_revision == revision:
            return
        key = frame_key(bgr)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self._present(cached, revision)
            self._pending = None
            return
        if self._job_token == (self._generation, revision):
            return
        # 仅一个待处理帧；翻页时立即替换，不积压过时的截图。
        self._pending = (self._generation, revision, key, bgr.copy())
        self._dispatch()

    @Slot()
    def _dispatch(self):
        if not self.running or self._job is not None or self._pending is None or self._retry.isActive():
            return
        generation, revision, key, bgr = self._pending
        if generation != self._generation or revision != self.tracker.revision:
            self._pending = None
            return
        self._pending = None
        ok, data = cv2.imencode(".png", bgr)
        if not ok:
            self.bridge._log("实时截图编码失败")
            return
        self._job_token, self._job_key = (generation, revision), key
        self.bridge._busy = True
        self.bridge.busy_changed.emit(True)
        # 实时模式用的是刚刚抓到的帧，不是框选时冻结的那张，提示词要说准。
        # 同时带上这一帧作原图对照（复用 captured_png，不重复编码）。
        self._job = TranslateWorker(self.bridge._pipeline, self._region, self,
                                    captured_png=data.tobytes(), source="实时截图",
                                    include_original=True)
        self._job.progress.connect(self._job_progress)
        self._job.done.connect(self._job_done)
        self._job.finished.connect(self._job_finished)
        self._job.start()

    @Slot(str)
    def _job_progress(self, message):
        if self.running and self._job_token == (self._generation, self.tracker.revision):
            self.bridge._log(message)

    @Slot(str)
    def _job_done(self, payload):
        try:
            data = json.loads(payload)
            if self.running and self._job_token == (self._generation, self.tracker.revision):
                if data.get("ok"):
                    if self._cache_limit:
                        self._cache[self._job_key] = data
                        self._cache.move_to_end(self._job_key)
                        while len(self._cache) > self._cache_limit:
                            self._cache.popitem(last=False)
                    self._present(data, self.tracker.revision)
                else:
                    self.bridge._log("实时翻译出错：" + data.get("error", "未知错误"))
                    self.bridge.result_ready.emit(payload)
                    self._retry.start()
        except Exception as error:
            self.bridge._log("实时结果处理失败：" + str(error))
            self._retry.start()

    def _present(self, data, revision):
        self._last_payload = data
        self._displayed_revision = revision
        self._remember_pairs(data)
        self.overlay.set_result(data)
        if not self.toolbar.hide_text_checkbox.isChecked() and self.running and not self._hidden_for_capture:
            self.overlay.show()
        self.bridge._log("实时完成：%d 个文字块" % len(data.get("boxes", [])))
        self.bridge.result_ready.emit(json.dumps(data, ensure_ascii=False))

    def _remember_pairs(self, data):
        """把这一页的译文并入会话记录，供「导出本次阅读」使用。

        按原文去重（首次出现优先）：同一句在翻回上一页时会重复识别，
        导出一份带重复的对照表反而干扰校对。
        """
        # 这里的结果来自后台线程序列化的 payload，格式不该出错；但一旦出错
        # 不能把整个阅读循环带崩，所以非预期结构一律当"没有内容"忽略。
        if not isinstance(data, dict):
            return
        pairs = data.get("pairs") or data.get("translations") or {}
        if isinstance(pairs, dict):
            items = list(pairs.items())
        elif isinstance(pairs, (list, tuple)):
            items = [(p.get("source", p.get("src", "")),
                      p.get("translation", p.get("target", p.get("dst", ""))))
                     if isinstance(p, dict) else (p[0], p[1])
                     for p in pairs if isinstance(p, (dict, list, tuple))
                     and (not isinstance(p, (list, tuple)) or len(p) >= 2)]
        else:
            return
        for source, target in items:
            source = str(source or "").strip()
            target = str(target or "").strip()
            if not source or not target or source in self._session_pairs:
                continue
            self._session_pairs[source] = target

    @property
    def session_pairs(self):
        return dict(self._session_pairs)

    @Slot()
    def _job_finished(self):
        job, self._job = self._job, None
        self._job_token = None
        self._job_key = None
        self.bridge._busy = False
        self.bridge.busy_changed.emit(False)
        if job is not None:
            job.deleteLater()
        self._dispatch()

    @Slot(str)
    def _capture_failed(self, error):
        self._restore_capture_windows()
        if self.running:
            self.pause()
            self.bridge._log("抓屏失败，实时已暂停：" + error)

    @Slot()
    def _capture_finished(self):
        capture, self._capture = self._capture, None
        self._restore_capture_windows()
        if capture is not None:
            capture.deleteLater()

    @Slot()
    def refresh(self):
        if not self.running:
            return
        self._generation += 1
        self.tracker.reset()
        self._cache.clear()
        self._pending = None
        self._displayed_revision = None
        self._last_payload = None
        self.overlay.clear_result()
        self.toolbar.set_status("正在重新识别…")

    @Slot()
    def show_settings(self):
        self.pause()
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    @Slot(float, str, bool, float)
    def _style_changed(self, scale, color, auto_color, stroke_ratio):
        """改字号/颜色/描边后立刻按新样式重排；重排要重跑 OCR，所以明确告诉用户在等什么。"""
        for target in (self.cfg, self._session_cfg):
            if isinstance(target, dict):
                render_cfg = target.setdefault("render", {})
                render_cfg["font_scale"] = float(scale)
                render_cfg["color"] = str(color)
                render_cfg["auto_color"] = bool(auto_color)
                render_cfg["stroke_ratio"] = float(stroke_ratio)
        if self.bridge._pipeline is not None and self._session_cfg is not None:
            self.bridge._pipeline.cfg = self._session_cfg
        detail = "自动配色" if auto_color else f"固定色 {color}"
        self.bridge._log(f"译文样式已更新：字号 ×{scale:.1f}、{detail}、描边 {stroke_ratio * 100:.0f}%")
        if self.running:
            self.refresh()
            self.bridge._log("正在按新样式重新识别并排版…")

    @Slot()
    def reselect(self):
        was_running = self.running
        self.pause()
        if self.bridge._busy:
            self.bridge._log("请等当前任务结束后重新框选")
            return
        self._resume_after_pick = was_running
        self.toolbar.hide()
        self.bridge.pick_region()

    @Slot(str)
    def _region_selected(self, text):
        self._region = parse_region(text)

    @Slot()
    def _selection_finished(self):
        self.toolbar.show()
        if self._resume_after_pick:
            self._resume_after_pick = False
            QTimer.singleShot(160, self.toggle)

    @Slot(bool)
    def _hide_text(self, hidden):
        if hidden:
            self.overlay.hide()
        elif self.running and not self._hidden_for_capture:
            self.overlay.show()

    @Slot(bool)
    def _show_original(self, show):
        """按住对照：只切换覆盖层画的是原图还是译文，不动已缓存的结果。"""
        self.overlay.set_show_original(show)

    @Slot(bool)
    def _toggle_marking(self, marking):
        """进入/退出标记模式：只切换覆盖层的点击接收，不打断翻译流程。"""
        self.overlay.set_marking(marking)
        if marking:
            # 标记与单句对照互斥：开一个就关掉另一个，避免点击意图混淆。
            if self.toolbar.peek_button.isChecked():
                self.toolbar.peek_button.setChecked(False)
            self.bridge._log("标记模式：点击译文气泡标记为「译得不好」，再点一次取消；导出时带上")
        else:
            self.bridge._log("已退出标记模式，覆盖层恢复点击穿透，可正常翻页")

    @Slot(bool)
    def _toggle_peeking(self, peeking):
        """进入/退出单句原图对照：点气泡只在那句盖回原图，不切整页。"""
        self.overlay.set_peeking(peeking)
        if peeking:
            if self.toolbar.mark_button.isChecked():
                self.toolbar.mark_button.setChecked(False)
            hint = "" if not self.overlay._original.isNull() else "（本页无原图帧，可能看不到效果）"
            self.bridge._log("单句对照：点击气泡显示该句原文，再点同一句或点别句切换" + hint)
        else:
            self.bridge._log("已退出单句对照，覆盖层恢复点击穿透")

    @Slot(str)
    def _on_sentence_clicked(self, source):
        """切换某句的标记并同步到覆盖层的标记框；同一句再点一次即取消。"""
        source = str(source or "").strip()
        if not source:
            return
        if source in self._marks:
            self._marks.pop(source, None)
            self.bridge._log(f"已取消标记：{source}")
        else:
            self._marks[source] = "译得不好"
            self.bridge._log(f"已标记为译得不好：{source}")
        self.overlay.set_marks(self._marks.keys())

    @Slot()
    def export_session(self):
        """导出本次阅读累积的译文对照。

        优先用会话累积（翻过的每一页都算），没有累积就退回最近一页，
        免得用户刚翻译完一页点导出却被告知"没内容"。
        """
        pairs = self.session_pairs
        source_label = "本次阅读（实时截图）"
        if not pairs:
            latest = (self._last_payload or {}).get("pairs") or \
                     (self._last_payload or {}).get("translations") or {}
            if latest:
                pairs = latest
                source_label = "当前页"
        if not pairs:
            self.bridge._log("还没有可导出的译文：先翻译一页，或先进入实时阅读")
            return
        if self.running:
            self.bridge._log("实时阅读进行中，先暂停再导出，避免内容还在变")
            return
        if self._marks:
            self.bridge._log(f"本次导出将带上 {len(self._marks)} 句标记")
        self.bridge.export_pairs(dict(pairs), source_label, dict(self._marks))

    @property
    def session_marks(self):
        """本次阅读累积的译错标记（原文 → 理由），供测试与外部查看。"""
        return dict(self._marks)

    @Slot()
    def leave_reading(self):
        """工具栏「退出」：结束透明阅读、回到设置页，不关闭程序。

        直接关窗会让人误以为丢掉了设置；回到设置页再决定是继续还是真退出。
        """
        was_running = self.running
        self.pause()
        self.toolbar.hide()
        self.overlay.hide()
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()
        self.bridge._log("已退出透明阅读，回到设置页"
                         + ("（刚才的识别已停止）" if was_running else "")
                         + "；要彻底退出请点设置页的「退出」")

    @Slot()
    def exit_requested(self):
        self.pause()
        if self.bridge._busy:
            self.bridge._log("后台任务尚未结束，请等完成后退出")
            return
        self.toolbar.hide()
        self.panel.close()
        self.closed.emit()

    @Slot(dict)
    def apply_hotkeys(self, bindings):
        """从设置页改绑快捷键；写入会话配置、立即重新注册，并写回 config.yaml。"""
        if self.bridge._busy or self.running:
            self.bridge._log("任务进行中，稍后再改快捷键")
            return
        clean = {str(k): (str(v).strip() if v else "")
                 for k, v in dict(bindings or {}).items()}
        self.cfg["hotkeys"] = clean
        self._hotkeys.rebind(clean)
        self._write_hotkeys_back(clean)

    def _write_hotkeys_back(self, bindings):
        """把改绑结果写回 config.yaml；没带配置路径或写失败时仅本次会话生效。"""
        path = str(self.cfg.get("_config_path") or "").strip()
        if not path:
            return
        try:
            changed = update_section_values(path, "hotkeys", bindings)
        except OSError as error:
            self.bridge._log(f"快捷键写回 config.yaml 失败（本次会话仍生效）：{error}")
            return
        if changed:
            self.bridge._log("快捷键已写回 config.yaml，重启后仍然生效")
        else:
            self.bridge._log("快捷键与 config.yaml 一致，无需写回")

    def shutdown(self):
        self._hotkeys.stop()
        self.pause()
        if self._capture is not None and self._capture.isRunning():
            self._capture.wait()
        if self._job is not None and self._job.isRunning():
            self._job.wait()
        self.overlay.close()
        self.toolbar.close()

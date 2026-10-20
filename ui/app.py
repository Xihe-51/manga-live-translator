# -*- coding: utf-8 -*-
"""GUI 默认使用原生设置页与透明阅读层；--web-panel 保留旧网页调试面板。"""
import os
import sys

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QMessageBox
from .bridge import Bridge, WarmupThread

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


def run_gui(cfg, web_panel=False):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    from .reading import NativePanel
    from .realtime import RealtimeController

    if web_panel:
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineWidgets import QWebEngineView
        base = QWebEngineView
    else:
        base = NativePanel

    class TranslationView(base):
        def closeEvent(self, event):
            if bridge._busy or warmup.isRunning():
                bridge._log("后台任务尚未结束，请等完成后再关闭窗口")
                event.ignore()
                return
            # 关窗即退出程序，先确认一次；透明阅读里点「退出」只是回设置页，不会走到这里。
            answer = QMessageBox.question(
                self, "退出漫画翻译", "确定要退出吗？\n退出后需要重新框选阅读区域。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            controller.shutdown()
            bridge.shutdown()
            super().closeEvent(event)
            app.quit()

    view = TranslationView() if web_panel else TranslationView(cfg)
    bridge = Bridge(cfg, panel=view)
    controller = RealtimeController(bridge, view, cfg)
    if web_panel:
        view.setWindowTitle("漫画翻译 · 网页调试面板")
        view.resize(1000, 720)
        channel = QWebChannel()
        channel.registerObject("bridge", bridge)
        view.page().setWebChannel(channel)
        view.load(QUrl.fromLocalFile(os.path.join(_WEB_DIR, "index.html")))
    else:
        view.pick_requested.connect(bridge.pick_region)
        view.single_requested.connect(bridge.translate_page)
        view.recognize_requested.connect(bridge.recognize_page)
        view.realtime_requested.connect(controller.toggle_region)
        view.provider_requested.connect(bridge.set_provider)
        view.inpaint_requested.connect(bridge.set_inpaint)
        view.source_language_requested.connect(bridge.set_source_language)
        view.grain_requested.connect(bridge.set_grain)
        view.prompts_requested.connect(bridge.set_prompts)
        view.hotkeys_requested.connect(controller.apply_hotkeys)
        view.test_connection_requested.connect(bridge.test_connection)
        view.model_requested.connect(bridge.set_model)
        view.preset_requested.connect(bridge.apply_preset)
        bridge.models_ready.connect(view.show_models_result)
        view.export_requested.connect(controller.export_session)
        # 设置页的「退出」也走 closeEvent，确认逻辑只写一处。
        view.exit_requested.connect(view.close)
        bridge.region_ready.connect(view.set_region)
        bridge.busy_changed.connect(view.set_busy)
        bridge.log_line.connect(view.append_log)
        bridge.result_ready.connect(view.show_result)
    warmup = WarmupThread(cfg)
    warmup.ready.connect(bridge.on_pipeline_ready)
    warmup.start()
    if not web_panel:
        # 快捷键要在窗口显示前就绪，用户不用等界面出来才生效。
        controller.start_hotkeys()
    view.show()
    # 强引用保留在 app.exec 的栈帧中，所有 QThread 在关闭前等待结束。
    sys.exit(app.exec())

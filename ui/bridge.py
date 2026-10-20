# -*- coding: utf-8 -*-
"""JS ↔ Python 桥；所有跨线程回调须声明固定 Slot，避免破坏 QWebChannel 信号索引。"""
import base64
import copy
import io
import json

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QThread, QTimer, Signal, Slot


def parse_region(text):
    """把 l,t,w,h 解析为物理像素区域。"""
    parts = str(text).strip().strip("[]()").replace("，", ",").split(",")
    if len(parts) != 4:
        return None
    try:
        vals = [int(p.strip()) for p in parts]
    except ValueError:
        return None
    return vals if vals[2] > 0 and vals[3] > 0 else None


def write_export(pairs, cfg, title=None, when=None, marks=None):
    """把译文对照写到文件，返回 (路径, 条数)。没内容可写时抛 ValueError。

    marks 是阅读时标记的「译得不好」句子（{原文: 理由} 或 set），会在导出文本
    里带上标记行，方便事后据此调提示词。单独抽成模块级函数是为了能在测试里直接调。
    """
    import os
    from engine import export as export_mod

    export_cfg = (cfg or {}).get("export", {}) or {}
    cleaned = export_mod.clean_pairs(pairs, drop_noise=bool(export_cfg.get("drop_noise", True)))
    if not cleaned:
        raise ValueError("没有可导出的译文（可能是还没翻译，或这一页只有符号碎片）")

    text = export_mod.render_pairs(cleaned, header=title, marks=marks,
                                   drop_noise=bool(export_cfg.get("drop_noise", True)))
    folder = str(export_cfg.get("dir") or "outputs")
    os.makedirs(folder, exist_ok=True)
    if export_cfg.get("timestamp_name", True):
        # 传 folder 让命名逻辑避开同名文件：时间戳只到秒，连点两次导出会撞名覆盖。
        name = export_mod.default_export_name(when=when, folder=folder)
    else:
        name = "译文对照.txt"
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path, len(cleaned)


class WarmupThread(QThread):
    ready = Signal(object)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._cfg = cfg

    def run(self):
        try:
            from engine import Pipeline
            self.ready.emit(Pipeline(self._cfg))
        except Exception as e:
            self.ready.emit(e)


class TranslateWorker(QThread):
    done = Signal(str)
    progress = Signal(str)
    captured = Signal()

    def __init__(self, pipeline, region, parent=None, captured_png=None,
                 source="框选时冻结的截图", recognize_only=False, include_original=False):
        super().__init__(parent)
        self._pipeline = pipeline
        self._region = list(region)
        self._captured_png = captured_png
        self._source = source
        self._recognize_only = bool(recognize_only)
        self._include_original = bool(include_original)

    def run(self):
        try:
            if self._captured_png is not None:
                import cv2
                import numpy as np
                self.progress.emit(f"正在读取{self._source}…")
                bgr = cv2.imdecode(np.frombuffer(self._captured_png, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError("框选截图读取失败，请重新截图")
            else:
                from engine.capture import ScreenCapturer
                self.progress.emit("正在截取屏幕区域…")
                cap = ScreenCapturer()
                try:
                    bgr = cap.grab(self._region)
                finally:
                    cap.close()
            self.captured.emit()
            if self._recognize_only:
                # 诊断模式：只识别、只标注，绝不抹字或翻译原图。
                image_pil, boxes, language = self._pipeline.recognize_only(
                    bgr, progress=self.progress.emit)
                self.progress.emit("正在生成识别标注预览并回传界面…")
                buf = io.BytesIO()
                image_pil.save(buf, format="PNG")
                payload = {"ok": True, "recognize_only": True,
                           "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                           "boxes": boxes, "translations": {}, "language": language}
            else:
                image_pil, boxes, translations = self._pipeline.process(bgr, progress=self.progress.emit)
                self.progress.emit("正在生成预览并回传界面…")
                buf = io.BytesIO()
                image_pil.save(buf, format="PNG")
                payload = {"ok": True, "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                           "boxes": boxes, "translations": translations}
                if self._include_original:
                    payload["original"] = self._original_data_url(bgr)
        except Exception as e:
            payload = {"ok": False, "error": str(e)}
        self.done.emit(json.dumps(payload, ensure_ascii=False))

    def _original_data_url(self, bgr):
        """原图对照帧：实时路径直接复用已编码 PNG，避免为同一帧重复编码。"""
        raw = self._captured_png
        if raw is None:
            import cv2
            ok, data = cv2.imencode(".png", bgr)
            if not ok:
                return ""
            raw = data.tobytes()
        return "data:image/png;base64," + base64.b64encode(bytes(raw)).decode()


class ModelsWorker(QThread):
    """后台 GET /models：连接测试 + 模型发现，结果（或错误说明）回传主线程。"""

    done = Signal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._cfg = cfg

    def run(self):
        import requests
        from engine.translate import fetch_models

        llm = self._cfg.get("llm", {}) or {}
        provider = llm.get("provider", "free")
        endpoint = llm.get(provider) or {}
        try:
            url, models = fetch_models(endpoint)
        except RuntimeError as error:
            payload = {"ok": False, "error": str(error)}
        except requests.exceptions.ConnectionError:
            payload = {"ok": False,
                       "error": f"连不上 {endpoint.get('base_url')}；"
                                "本地服务请确认已启动，云端请检查网络"}
        except requests.exceptions.Timeout:
            payload = {"ok": False, "error": "连接测试超时，请检查服务地址或网络"}
        except requests.exceptions.HTTPError as error:
            status = error.response.status_code if error.response is not None else "?"
            if status == 401:
                hint = "鉴权失败，请检查该通道的 api_key"
            elif status == 404:
                hint = "该服务没有 /models 接口，无法自动发现模型；可直接手填模型名"
            else:
                hint = f"接口返回 HTTP {status}"
            payload = {"ok": False, "error": hint}
        except (KeyError, TypeError, ValueError):
            payload = {"ok": False, "error": "接口返回的模型列表格式不正确"}
        else:
            payload = {"ok": True, "url": url, "provider": provider,
                       "models": models, "current": endpoint.get("model")}
        self.done.emit(json.dumps(payload, ensure_ascii=False))


class Bridge(QObject):
    log_line = Signal(str)
    result_ready = Signal(str)
    region_ready = Signal(str)
    busy_changed = Signal(bool)
    model_ready = Signal(bool)
    selection_finished = Signal()
    reading_mode_changed = Signal(bool)
    models_ready = Signal(str)

    def __init__(self, cfg, parent=None, panel=None):
        super().__init__(parent)
        self._cfg = cfg
        self._panel = panel
        self._panel_hidden = False
        self._pipeline = None
        self._worker = None
        self._models_worker = None
        self._selector = None
        self._picking = False
        self._busy = False
        self._reading_active = False
        self._selected_region = None
        self._selected_png = None
        self._pending_region = None
        self._pending_cfg = None
        self._recognize_only = False

    @Slot(object)
    def on_pipeline_ready(self, obj):
        if isinstance(obj, Exception):
            self._log("模型加载失败：" + str(obj))
            self.model_ready.emit(False)
        else:
            self._pipeline = obj
            self._log("OCR 模型已就绪，可以截图并翻译了")
            self.model_ready.emit(True)

    def _log(self, message):
        print(message, flush=True)
        self.log_line.emit(message)

    @Slot(str)
    def set_provider(self, provider):
        if provider in ("free", "paid", "mock") and not self._busy and not self._reading_active:
            self._cfg["llm"]["provider"] = provider
            self._log(f"翻译通道已切换为 {provider}")

    @Slot(str)
    def set_inpaint(self, method):
        if method in ("cover", "cv2", "lama") and not self._busy and not self._reading_active:
            self._cfg.setdefault("inpaint", {})["method"] = method
            hint = "（每页约 2 秒，背景纹理更干净）" if method == "lama" else ""
            self._log(f"抹字方式已切换为 {method}{hint}")

    @Slot()
    def test_connection(self):
        """测试当前通道连接并发现可用模型（后台线程 GET /models）。"""
        if self._busy or self._picking or self._reading_active:
            self._log("当前任务尚未结束，请稍后再测连接")
            return
        if self._models_worker is not None:
            self._log("连接测试还在进行中，请稍候…")
            return
        provider = self._cfg.get("llm", {}).get("provider", "free")
        self._log(f"正在测试 {provider} 通道连接并获取模型列表…")
        self._models_worker = ModelsWorker(copy.deepcopy(self._cfg), self)
        self._models_worker.done.connect(self._on_models_ready)
        self._models_worker.finished.connect(self._on_models_finished)
        self._models_worker.start()

    @Slot(str)
    def _on_models_ready(self, payload):
        try:
            data = json.loads(payload)
            if data.get("ok"):
                self._log(f"连接成功：{data.get('url')}，发现 {len(data.get('models', []))} 个模型")
            else:
                self._log("连接测试失败：" + str(data.get("error", "未知错误")))
        except (TypeError, ValueError):
            pass
        self.models_ready.emit(payload)

    @Slot()
    def _on_models_finished(self):
        worker, self._models_worker = self._models_worker, None
        if worker is not None:
            worker.deleteLater()

    @Slot(str)
    def set_model(self, model):
        """把选中的模型写回当前通道（本次会话生效）。"""
        model = str(model or "").strip()
        if not model:
            return
        if self._busy or self._reading_active:
            self._log("任务进行中，暂不能切换模型")
            return
        provider = self._cfg["llm"].get("provider", "free")
        self._cfg["llm"].setdefault(provider, {})["model"] = model
        self._log(f"{provider} 通道模型已切换为 {model}（本次会话生效）")

    @Slot(dict)
    def apply_preset(self, preset):
        """套用供应商预设：写通道的 base_url/model，并整体替换 llm.options。"""
        if not isinstance(preset, dict):
            return
        if self._busy or self._reading_active:
            self._log("任务进行中，暂不能套用预设")
            return
        provider = str(preset.get("provider") or "").strip()
        if provider not in ("free", "paid"):
            self._log("预设无效：provider 只能是 free 或 paid，已忽略")
            return
        llm = self._cfg.setdefault("llm", {})
        name = str(preset.get("name") or provider).strip()
        channel = llm.get(provider)
        if not isinstance(channel, dict):
            channel = llm[provider] = {}
        base_url = str(preset.get("base_url") or "").strip()
        model = str(preset.get("model") or "").strip()
        if base_url:
            channel["base_url"] = base_url
        if model:
            channel["model"] = model
        options = preset.get("options")
        if isinstance(options, dict):
            llm["options"] = copy.deepcopy(options)
        llm["provider"] = provider
        hint = f"，模型 {channel.get('model')}" if channel.get("model") else ""
        self._log(f"已套用预设「{name}」：{provider} 通道{hint}（本次会话生效）")

    @Slot(str)
    def set_source_language(self, language):
        if language in ("auto", "ja", "en", "ko", "ru") and not self._busy and not self._reading_active:
            from engine.translate import language_name
            self._cfg.setdefault("llm", {})["source_lang"] = language
            label = "自动识别" if language == "auto" else language_name(language)
            self._log(f"原文语言已切换为 {label}（本次会话生效）")

    @Slot(bool)
    def set_grain(self, enabled):
        if not self._busy and not self._reading_active:
            self._cfg.setdefault("inpaint", {})["grain"] = bool(enabled)
            self._log("随机颗粒已" + ("开启（实验选项，不是网点重建）" if enabled else "关闭"))

    @Slot(str)
    def recognize_page(self, region_text):
        """只识别不翻译：用当前选区跑一遍 OCR，回传带编号标注的原图。"""
        if self._busy or self._picking or self._reading_active:
            self._log("当前操作尚未结束，请稍候…")
            return
        region = parse_region(region_text)
        if region is None:
            self._on_done(json.dumps({"ok": False, "error": "区域格式错误：应为 左,上,宽,高"}, ensure_ascii=False))
            return
        if self._pipeline is None:
            self._log("OCR 模型还在加载，请稍候再点")
            return
        self._busy = True
        self.busy_changed.emit(True)
        self._pending_region = region
        self._pending_cfg = copy.deepcopy(self._cfg)
        self._recognize_only = True
        self._log(f"只识别不翻译：区域 {region}（不会抹字、不会调用翻译）")
        if self._selected_png is not None and region == self._selected_region:
            self._start_translation()
        else:
            self._hide_panel()
            QTimer.singleShot(150, self._start_translation)

    @Slot(str, str)
    def set_prompts(self, system, batch):
        """更新翻译提示词模板；空串表示恢复内置默认。"""
        if self._busy or self._reading_active:
            self._log("任务进行中，暂不能改提示词")
            return
        prompts = self._cfg.setdefault("llm", {}).setdefault("prompts", {})
        prompts["system"] = system or None
        prompts["batch"] = batch or None
        scope = []
        if system:
            scope.append("系统")
        if batch:
            scope.append("批量")
        self._log("提示词已更新：" + ("、".join(scope) + "（本次会话生效）" if scope
                                     else "已恢复内置默认"))

    @Slot(dict, str, object)
    def export_pairs(self, pairs, source="", marks=None):
        """把译文对照写成文本文件。用当前会话的配置决定目录与命名。"""
        try:
            path, count = write_export(pairs, self._cfg, marks=marks,
                                       title=(f"来源：{source}" if source else None))
        except ValueError as error:
            self._log("导出未完成：" + str(error))
            return
        except OSError as error:
            self._log(f"导出失败：无法写入文件（{error}）")
            return
        # 路径要写全，用户才知道去哪找；相对路径在打包后会变成意外位置。
        import os
        self._log(f"已导出 {count} 条译文对照 -> {os.path.abspath(path)}")

    @Slot(str)
    def translate_page(self, region_text):
        if self._busy or self._picking or self._reading_active:
            self._log("当前操作尚未结束，请稍候…")
            return
        region = parse_region(region_text)
        if region is None:
            self._on_done(json.dumps({"ok": False, "error": "区域格式错误：应为 左,上,宽,高"}, ensure_ascii=False))
            return
        if self._pipeline is None:
            self._log("OCR 模型还在加载，请稍候再点")
            return
        self._busy = True
        self.busy_changed.emit(True)
        self._pending_region = region
        # 固定本次配置，避免处理中切换通道影响正在运行的任务。
        self._pending_cfg = copy.deepcopy(self._cfg)
        self._recognize_only = False
        self._log(f"开始翻译区域 {region} …")
        if self._selected_png is not None and region == self._selected_region:
            self._start_translation()
        else:
            # 手填坐标为实时抓屏；先隐藏面板，防止抓到自己的操作界面。
            self._hide_panel()
            QTimer.singleShot(150, self._start_translation)

    @Slot()
    def _start_translation(self):
        try:
            region = self._pending_region
            frozen = self._selected_png if region == self._selected_region else None
            # Pipeline 常驻，期间不接受切换通道或新任务。
            self._pipeline.cfg = self._pending_cfg
            self._pipeline.translator.cfg = self._pending_cfg
            self._worker = TranslateWorker(self._pipeline, region, self, captured_png=frozen,
                                           recognize_only=self._recognize_only)
            self._worker.progress.connect(self._on_progress)
            self._worker.captured.connect(self._on_captured)
            self._worker.done.connect(self._on_done)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.start()
        except Exception as e:
            self._on_done(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
            self._busy = False
            self.busy_changed.emit(False)
            self._restore_panel()

    @Slot()
    def pick_region(self):
        if self._picking or self._busy or self._reading_active:
            self._log("当前操作尚未结束，请稍候…")
            return
        self._picking = True
        self.busy_changed.emit(True)
        self._hide_panel()
        # 槽立即返回，让 QWebChannel 正常处理后续信号，不在槽内跑嵌套事件循环。
        QTimer.singleShot(150, self._show_selector)

    @Slot()
    def _show_selector(self):
        try:
            from .region_selector import RegionSelector
            from PySide6.QtGui import QCursor, QGuiApplication
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            self._selector = RegionSelector(screen=screen)
            self._selector.region_selected.connect(self._on_region_selected)
            self._selector.cancelled.connect(self._on_region_cancelled)
            self._selector.show()
            self._selector.raise_()
            self._selector.activateWindow()
        except Exception as e:
            self._log("截图失败：" + str(e))
            self._finish_selection()

    @Slot(list)
    def _on_region_selected(self, region):
        try:
            image = self._selector.selected_image()
            data = QByteArray()
            buf = QBuffer(data)
            buf.open(QIODevice.WriteOnly)
            if image.isNull() or not image.save(buf, "PNG"):
                raise RuntimeError("框选截图为空，请重新拖动框选")
            buf.close()
            self._selected_region = list(region)
            self._selected_png = bytes(data)
            self.region_ready.emit(",".join(str(v) for v in region))
            self._log("截图已冻结；翻译会使用刚才选中的画面，不会重新截到其他窗口")
        except Exception as e:
            self._log("截图失败：" + str(e))
        finally:
            self._finish_selection()

    @Slot()
    def _on_region_cancelled(self):
        self._log("已取消截图，原选区保留")
        self._finish_selection()

    def _finish_selection(self):
        selector, self._selector = self._selector, None
        if selector is not None:
            selector.close()
            selector.deleteLater()
        self._picking = False
        self.busy_changed.emit(False)
        self._restore_panel()
        self.selection_finished.emit()

    def _hide_panel(self):
        if self._panel is not None and self._panel.isVisible():
            self._panel_hidden = True
            self._panel.hide()

    def _restore_panel(self):
        if self._panel is not None and self._panel_hidden:
            self._panel_hidden = False
            self._panel.show()
            self._panel.raise_()
            self._panel.activateWindow()

    @Slot(result=str)
    def get_config(self):
        public = copy.deepcopy(self._cfg)
        for options in public.get("llm", {}).values():
            if isinstance(options, dict):
                options.pop("api_key", None)
        return json.dumps(public, ensure_ascii=False)

    @Slot()
    def _on_captured(self):
        self._restore_panel()

    @Slot(str)
    def _on_progress(self, message):
        self._log(message)

    @Slot(str)
    def _on_done(self, payload):
        try:
            data = json.loads(payload)
            if data.get("ok"):
                count = len(data.get("boxes", []))
                self._log(f"完成：共 {count} 个文字块" if count else "完成：未检测到文字；请重新截图，确认选区包含清晰日文")
            else:
                self._log("出错：" + data.get("error", "未知错误"))
            self.result_ready.emit(payload)
        except Exception as e:
            self._log("结果处理失败：" + str(e))
        self._restore_panel()

    @Slot()
    def _on_worker_finished(self):
        self._busy = False
        self.busy_changed.emit(False)

    def shutdown(self):
        """关闭时等待后台线程，避免 QThread still running 导致进程异常退出。"""
        if self._selector is not None:
            self._selector.close()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        if self._models_worker is not None and self._models_worker.isRunning():
            self._models_worker.wait()

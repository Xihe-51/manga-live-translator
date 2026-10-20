# -*- coding: utf-8 -*-
"""透明阅读视图及原生控制面板；只发出请求，不抓屏、不运行翻译。"""
import base64
import binascii
import ctypes
from ctypes import wintypes
import json
import math
import sys

from PySide6.QtCore import QEvent, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen, QPixmap, QScreen
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from engine.hotkey import format_hotkey, format_poll_key


WDA_EXCLUDEFROMCAPTURE = 0x11


def exclude_from_capture(widget) -> bool:
    """只在真实 Windows Qt 平台调用并核对显示亲和性，失败如实返回 False。"""
    if sys.platform != "win32" or QGuiApplication.platformName().lower() != "windows":
        return False
    try:
        hwnd = wintypes.HWND(int(widget.winId()))
        if not hwnd.value:
            return False
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        set_affinity = user32.SetWindowDisplayAffinity
        set_affinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        set_affinity.restype = wintypes.BOOL
        get_affinity = user32.GetWindowDisplayAffinity
        get_affinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        get_affinity.restype = wintypes.BOOL
        if not set_affinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            return False
        actual = wintypes.DWORD()
        return bool(get_affinity(hwnd, ctypes.byref(actual))) and actual.value == WDA_EXCLUDEFROMCAPTURE
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, ctypes.ArgumentError):
        return False


class _CaptureExcludedWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.capture_excluded = False

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.WinIdChange:
            # 不在构造阶段抢先创建窗口；首次创建或句柄重建后立即核对。
            self.capture_excluded = bool(self.internalWinId()) and exclude_from_capture(self)
        return result


def _decode_image(value):
    """只解码内存中的图像，不接受文件路径或远程地址。"""
    try:
        if isinstance(value, str):
            if value.startswith("data:"):
                header, separator, value = value.partition(",")
                if not separator or not header.startswith("data:image/") or ";base64" not in header:
                    return QImage()
            value = base64.b64decode(value, validate=True)
        if not isinstance(value, (bytes, bytearray)):
            return QImage()
        return QImage.fromData(bytes(value))
    except (ValueError, TypeError, binascii.Error):
        return QImage()


class ReadingOverlay(_CaptureExcludedWidget):
    #: 标记模式下点中某句时发出该句原文，供控制器切换标记。
    sentence_clicked = Signal(str)

    #: 新回填淡入的时长（毫秒）；够短到不拖沓，够长到眼睛能跟上。
    FADE_MS = 260

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._patch = QImage()
        self._original = QImage()
        self._show_original = False
        self._region = None
        self._screen = None
        self._dpr = 1.0
        # 当前页每句的框（图像坐标 + 原文），点击模式下据此把点击映射到句子。
        self._sentences = []
        self._image_size = None
        self._marks = set()
        self._mark_mode = False
        self._peek_mode = False
        # 单句原图对照：命中的框（图像坐标），画在译文之上。
        self._peek_rect = None
        # 新回填淡入：0→1 的透明度，外加一段同步衰减的高亮。
        self._fade = 1.0
        self._anim_elapsed = 0
        self._anim_total = 0
        reading = cfg.get("reading", {}) or {}
        self._fade_enabled = bool(reading.get("fade_in", True))
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._border_visible = bool(cfg.get("reading", {}).get("border_visible", False))
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                            | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAutoFillBackground(False)
        self.setFocusPolicy(Qt.NoFocus)

    def set_region(self, region, screen=None):
        try:
            x, y, width, height = map(float, region)
        except (TypeError, ValueError) as exc:
            raise ValueError("区域应为物理像素 [左, 上, 宽, 高]") from exc
        if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
            raise ValueError("区域坐标必须有限，宽高必须为正")
        physical = QRectF(x, y, width, height)
        selected = None
        for candidate in ([screen] if screen is not None else QGuiApplication.screens()):
            geometry = candidate.geometry()
            dpr = float(candidate.devicePixelRatio())
            bounds = QRectF(geometry.x(), geometry.y(), geometry.width() * dpr, geometry.height() * dpr)
            if bounds.contains(physical):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError("透明阅读仅支持单屏区域，请重新框选，不能跨屏")
        origin = selected.geometry().topLeft()
        dpr = float(selected.devicePixelRatio())
        # 屏幕原点是原生全局坐标，只将屏幕内偏移和尺寸除以 DPR。
        left = round(origin.x() + (x - origin.x()) / dpr)
        top = round(origin.y() + (y - origin.y()) / dpr)
        right = round(origin.x() + (x + width - origin.x()) / dpr)
        bottom = round(origin.y() + (y + height - origin.y()) / dpr)
        if isinstance(selected, QScreen):
            self.winId()
            self.windowHandle().setScreen(selected)
        self._screen, self._dpr = selected, dpr
        self._region = (x, y, width, height)
        self.setGeometry(left, top, max(1, right - left), max(1, bottom - top))
        self.clear_result()

    def set_result(self, payload_dict):
        self.clear_result()
        if not isinstance(payload_dict, dict) or payload_dict.get("ok") is False:
            return
        image = _decode_image(payload_dict.get("image"))
        if image.isNull():
            return
        boxes = payload_dict.get("boxes", [])
        sources = payload_dict.get("source_boxes", [])
        boxes = boxes if isinstance(boxes, (list, tuple)) else []
        sources = list(sources) if isinstance(sources, (list, tuple)) else []
        for box in boxes:
            if isinstance(box, dict) and isinstance(box.get("source_boxes"), (list, tuple)):
                sources.extend(box["source_boxes"])
        self._image_size = image.size()
        self._capture_sentences(boxes)
        # 合并排版框也必须保留：中文可画到原始竖列间隙，不能只截各 source_boxes。
        patch = QImage(image.size(), QImage.Format_RGBA8888)
        patch.fill(Qt.transparent)
        opaque = image.convertToFormat(QImage.Format_RGB32)
        pad = max(0, float(self._cfg.get("inpaint", {}).get("pad", 3)))
        painter = QPainter(patch)
        for box in [*sources, *boxes]:
            if not isinstance(box, dict):
                continue
            try:
                x, y, width, height = (float(box[key]) for key in ("x", "y", "w", "h"))
                if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
                    continue
                left, top = max(0, math.floor(x - pad)), max(0, math.floor(y - pad))
                right = min(image.width(), math.ceil(x + width + pad))
                bottom = min(image.height(), math.ceil(y + height + pad))
            except (KeyError, TypeError, ValueError):
                continue
            if right > left and bottom > top:
                rect = QRect(left, top, right - left, bottom - top)
                painter.drawImage(rect, opaque, rect)
        painter.end()
        self._patch = patch
        # 原图对照帧随结果一起保存；缺省或解码失败时按住只会看到译文 patch。
        self._original = _decode_image(payload_dict.get("original"))
        self._start_fade()
        self.update()

    def clear_result(self):
        self._patch = QImage()
        self._original = QImage()
        self._sentences = []
        self._image_size = None
        self._peek_rect = None
        self.update()

    # ---------- 新回填淡入 ----------
    def _start_fade(self):
        """新一页回填后从低透明度淡入，并短暂高亮新气泡，帮眼睛定位。"""
        if not self._fade_enabled:
            self._fade = 1.0
            self.update()
            return
        self._anim_elapsed = 0
        self._anim_total = self.FADE_MS
        self._fade = 0.0
        self._anim_timer.start()

    def _on_anim_tick(self):
        self._anim_elapsed += self._anim_timer.interval()
        if self._anim_total <= 0:
            self._anim_timer.stop()
            self._fade = 1.0
        else:
            progress = min(1.0, self._anim_elapsed / self._anim_total)
            # 缓出：起步快、收尾慢，比线性更自然。
            self._fade = 1.0 - (1.0 - progress) ** 2
            if progress >= 1.0:
                self._fade = 1.0
                self._anim_timer.stop()
        self.update()

    def _capture_sentences(self, boxes):
        """记下当前页每句的框与原文，供标记模式下按点击位置命中句子。"""
        sentences = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            text = str(box.get("text") or "").strip()
            if not text:
                continue
            try:
                x, y, width, height = (float(box[key]) for key in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in (x, y, width, height)) and width > 0 and height > 0:
                sentences.append((x, y, width, height, text))
        self._sentences = sentences

    def _hit(self, point):
        """把覆盖层内的一个点映射到图像坐标，返回命中的句框；没命中返回 None。

        图像被整体缩放到控件矩形，所以按比例换算即可；多框相接时取面积最小的
        那个（嵌套的合并框 vs 原句框），保证点到的是最具体的一句。
        """
        if self._image_size is None or self._image_size.isEmpty():
            return None
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0:
            return None
        x = point.x() * self._image_size.width() / width
        y = point.y() * self._image_size.height() / height
        hit, best = None, None
        for sentence in self._sentences:
            bx, by, bw, bh = sentence[:4]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                area = bw * bh
                if best is None or area < best:
                    best, hit = area, sentence
        return hit

    def sentence_at(self, point):
        """返回命中句子的原文；没命中返回空串。"""
        hit = self._hit(point)
        return hit[4] if hit is not None else ""

    def box_at(self, point):
        """返回命中句子的框 (x,y,w,h)；没命中返回 None。"""
        hit = self._hit(point)
        return hit[:4] if hit is not None else None

    def _update_input(self):
        """按当前点击模式决定本窗口是否接收鼠标；两个模式都不开时保持点击穿透。"""
        interactive = self._mark_mode or self._peek_mode
        self.setAttribute(Qt.WA_TransparentForMouseEvents, not interactive)
        self.setWindowFlag(Qt.WindowTransparentForInput, not interactive)
        self.setCursor(Qt.CrossCursor if interactive else Qt.ArrowCursor)
        if self.isVisible():
            # 改窗口标志会让窗口被隐藏，需要再显示一次才生效。
            self.show()
        self.update()

    def set_marking(self, marking):
        """进入/退出标记模式；与单句对照互斥，同一时间只接收一种点击意图。"""
        marking = bool(marking)
        if marking == self._mark_mode:
            return
        self._mark_mode = marking
        if marking:
            self._peek_mode = False
            self._peek_rect = None
        self._update_input()

    def set_peeking(self, peeking):
        """进入/退出单句原图对照；与标记模式互斥。退出时清掉已选中的框。"""
        peeking = bool(peeking)
        if peeking == self._peek_mode:
            return
        self._peek_mode = peeking
        if not peeking:
            self._peek_rect = None
        else:
            self._mark_mode = False
        self._update_input()

    def set_marks(self, sources):
        """更新已标记的句子集合，画出标记框；原文匹配，跨页仍显示。"""
        marks = {str(s).strip() for s in (sources or ()) if str(s).strip()}
        if marks != self._marks:
            self._marks = marks
            self.update()

    def _rect_for(self, box):
        """把图像坐标的框换算成覆盖层控件坐标的矩形。"""
        if self._image_size is None or self._image_size.isEmpty():
            return None
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0:
            return None
        x, y, bw, bh = box
        return QRectF(x * width / self._image_size.width(),
                      y * height / self._image_size.height(),
                      bw * width / self._image_size.width(),
                      bh * height / self._image_size.height())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._mark_mode:
            text = self.sentence_at(event.position())
            if text:
                self.sentence_clicked.emit(text)
                event.accept()
                return
        elif event.button() == Qt.LeftButton and self._peek_mode:
            box = self.box_at(event.position())
            if box is not None:
                # 再点同一句收起、点另一句切换；单句级别来回看，不切整页对照。
                self._peek_rect = None if self._peek_rect == box else box
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def set_show_original(self, show):
        """按住对照：只切换绘制源，不动已缓存的译文结果。"""
        self._show_original = bool(show)
        self.update()

    def set_border_visible(self, visible):
        self._border_visible = bool(visible)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        showing_original = self._show_original and not self._original.isNull()
        source = self._original if showing_original else self._patch
        # 译文视图按淡入进度透明绘制；按住看原图时始终不透明，方便核对。
        painter.setOpacity(1.0 if showing_original else self._fade)
        if not source.isNull():
            painter.drawImage(QRectF(self.rect()), source, QRectF(source.rect()))
        painter.setOpacity(1.0)
        # 新回填高亮：随淡入进度同步衰减，翻页后一眼找到刚出现的气泡。
        if not showing_original and self._fade < 1.0:
            highlight = int((1.0 - self._fade) * 90)
            if highlight > 0:
                for sentence in self._sentences:
                    rect = self._rect_for(sentence[:4])
                    if rect is not None:
                        painter.fillRect(rect, QColor(90, 170, 255, highlight))
        # 单句原图对照：把原图对应裁剪画在译文之上，只盖住这一句。
        if self._peek_rect is not None and not self._original.isNull():
            rect = self._rect_for(self._peek_rect)
            if rect is not None:
                painter.drawImage(rect, self._original,
                                  QRectF(*self._peek_rect))
                painter.setPen(QPen(QColor(40, 200, 120), 2))
                painter.drawRect(rect)
        # 标记框只在译文视图上画：按住看原图时让画面保持干净，便于核对抹字。
        if not showing_original:
            for bx, by, bw, bh, text in self._sentences:
                if text not in self._marks:
                    continue
                rect = self._rect_for((bx, by, bw, bh))
                if rect is None:
                    continue
                painter.fillRect(rect, QColor(255, 80, 80, 40))
                painter.setPen(QPen(QColor(230, 60, 60), 2, Qt.DashLine))
                painter.drawRect(rect)
        if self._border_visible:
            painter.setPen(QPen(QColor(55, 145, 255), 2))
            painter.drawRect(QRectF(self.rect()).adjusted(1, 1, -1, -1))
        elif self._mark_mode:
            # 标记模式给整页描一圈琥珀色虚线，提示此时点击是在标记而不是翻页。
            painter.setPen(QPen(QColor(255, 190, 60), 2, Qt.DashLine))
            painter.drawRect(QRectF(self.rect()).adjusted(1, 1, -1, -1))
        elif self._peek_mode:
            # 单句对照模式描一圈绿色虚线，提示点击是在看原图。
            painter.setPen(QPen(QColor(40, 200, 120), 2, Qt.DashLine))
            painter.drawRect(QRectF(self.rect()).adjusted(1, 1, -1, -1))
        painter.end()


_DARK_QSS = """
QWidget { background: #20232b; color: #e6e9ef; font-family: "Microsoft YaHei"; font-size: 12px; }
QPushButton { background: #343a48; border: 1px solid #495164; border-radius: 4px; padding: 5px 8px; }
QPushButton:hover { background: #45516a; }
QPushButton:pressed { background: #286bc1; }
QPushButton:disabled { background: #292e38; color: #737b8a; border-color: #353b47; }
QLineEdit, QComboBox, QPlainTextEdit { background: #171a21; border: 1px solid #414959; border-radius: 3px; padding: 4px; }
QLineEdit:disabled, QComboBox:disabled { color: #737b8a; }
QCheckBox { spacing: 4px; }
QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #68758c; border-radius: 2px; background: #171a21; }
QCheckBox::indicator:checked { background: #3786e8; border-color: #70b1ff; }
QScrollArea { border: 1px solid #414959; }
"""


class _DragLabel(QLabel):
    def __init__(self, text, parent):
        super().__init__(text, parent)
        self._offset = None
        self.setCursor(Qt.SizeAllCursor)
        self.setToolTip("拖动此标签移动工具栏")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._offset = event.globalPosition().toPoint() - self.window().pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._offset is not None and event.buttons() & Qt.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._offset)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._offset = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ReadingToolbar(_CaptureExcludedWidget):
    toggle_requested = Signal()
    region_requested = Signal()
    settings_requested = Signal()
    exit_requested = Signal()
    refresh_requested = Signal()
    export_requested = Signal()
    border_changed = Signal(bool)
    hide_text_changed = Signal(bool)
    original_hold = Signal(bool)      # 按住看原图：按下 True、松开 False
    mark_requested = Signal(bool)     # 标记模式开关：进入 True、退出 False
    peek_requested = Signal(bool)     # 单句原图对照开关：进入 True、退出 False
    style_changed = Signal(float, str, bool, float)   # 字号缩放、字色、自动配色、描边比例

    #: 固定宽度，够放「阅读 开始 重选 刷新 导出 标记 看原文 原图 样式 设置 退出 框线 隐藏译文 状态」。
    DEFAULT_WIDTH = 1000

    def __init__(self, cfg=None):
        super().__init__()
        self._running = False
        self._ready = False
        reading = (cfg or {}).get("reading", {})
        render_cfg = (cfg or {}).get("render", {}) or {}
        hotkeys = (cfg or {}).get("hotkeys", {}) or {}
        try:
            self._font_scale = float(render_cfg.get("font_scale", 1.0))
        except (TypeError, ValueError):
            self._font_scale = 1.0
        self._color = str(render_cfg.get("color", "#000000") or "#000000")
        self._auto_color = bool(render_cfg.get("auto_color", True))
        try:
            self._stroke_ratio = float(render_cfg.get("stroke_ratio", 0.08))
        except (TypeError, ValueError):
            self._stroke_ratio = 0.08
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                            | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet(_DARK_QSS)
        self.setWindowTitle("透明阅读工具栏")
        self.setFixedWidth(self.DEFAULT_WIDTH)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(5)
        self.drag_label = _DragLabel("阅读", self)
        layout.addWidget(self.drag_label)
        self.toggle_button = QPushButton("开始")
        self.region_button = QPushButton("重选")
        self.refresh_button = QPushButton("刷新")
        self.export_button = QPushButton("导出")
        self.style_button = QPushButton("样式")
        self.settings_button = QPushButton("设置")
        self.exit_button = QPushButton("退出")
        for button, signal in ((self.toggle_button, self.toggle_requested),
                               (self.region_button, self.region_requested),
                               (self.refresh_button, self.refresh_requested),
                               (self.export_button, self.export_requested),
                               (self.settings_button, self.settings_requested),
                               (self.exit_button, self.exit_requested)):
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda checked=False, target=signal: target.emit())
        self.style_button.setFocusPolicy(Qt.NoFocus)
        self.style_button.setToolTip("调整译文字号与颜色，立即按新样式重新排版")
        self.style_button.clicked.connect(self.open_style_dialog)
        # 按住型按钮：pressed/released 成对发出，与「按住对照键」行为一致。
        self.original_button = QPushButton("原图")
        self.original_button.setFocusPolicy(Qt.NoFocus)
        self.original_button.pressed.connect(lambda: self.original_hold.emit(True))
        self.original_button.released.connect(lambda: self.original_hold.emit(False))
        hold_label = ""
        try:
            hold_label = format_poll_key(reading.get("original_key"))
        except Exception:
            hold_label = ""
        self.original_button.setToolTip(
            "按住显示原图、松开恢复译文，用来核对抹字与排版有没有盖住原文。\n"
            + (f"快捷键：按住 {hold_label}" if hold_label else
               "快捷键：未配置（可在 config.yaml 的 reading.original_key 设置）"))
        # 标记按钮：开关式。进入后覆盖层才接收点击，点中译文气泡即标记为「译得不好」。
        self.mark_button = QPushButton("标记")
        self.mark_button.setCheckable(True)
        self.mark_button.setFocusPolicy(Qt.NoFocus)
        self.mark_button.toggled.connect(self.mark_requested.emit)
        self.mark_button.setToolTip(
            "开启标记模式后，点击译文气泡把它标记为「译得不好」，导出时会带上标记行。\n"
            "再次点击同一句取消标记；关闭标记模式后覆盖层恢复点击穿透，不影响翻页。")
        # 看原文按钮：开关式。进入后点某个气泡，只在那句位置盖回原图，单句级别对照。
        self.peek_button = QPushButton("看原文")
        self.peek_button.setCheckable(True)
        self.peek_button.setFocusPolicy(Qt.NoFocus)
        self.peek_button.toggled.connect(self.peek_requested.emit)
        self.peek_button.setToolTip(
            "开启后点击译文气泡，只在该句位置盖回原图，单句级别来回核对翻译。\n"
            "再点同一句收起、点别句切换；与「标记」互斥，关闭后恢复点击穿透。")
        # 工具栏在阅读时是唯一能摸到的 UI，把快捷键写进 tooltip 才不用去翻设置页。
        def with_hotkey(text, action, extra=""):
            combo = str(hotkeys.get(action) or "").strip()
            if not combo:
                return text + extra
            try:
                combo = format_hotkey(combo)
            except Exception:
                pass
            return f"{text}\n快捷键：{combo}{extra}"

        self.toggle_button.setToolTip(with_hotkey("开始或暂停透明实时阅读", "toggle"))
        self.region_button.setToolTip(with_hotkey("重新框选漫画阅读区域", "reselect"))
        self.refresh_button.setToolTip(with_hotkey("重新识别当前页", "refresh"))
        self.export_button.setToolTip(
            "导出译文对照：本次阅读翻过的每一页都会累积，原句/译句两行一组。\n"
            "被「标记」的句子会带上标记行，方便事后据此调提示词。\n"
            "导出后运行日志会给出完整文件路径。")
        self.settings_button.setToolTip(with_hotkey("收起阅读层，回到设置页", "settings"))
        self.exit_button.setToolTip(with_hotkey("退出透明阅读，回到设置页", "leave"))
        # 按阅读顺序加进布局：标记/看原文紧挨导出，核对完顺手就能标记或回看原图。
        for widget in (self.toggle_button, self.region_button, self.refresh_button,
                       self.export_button, self.mark_button, self.peek_button,
                       self.original_button, self.style_button, self.settings_button,
                       self.exit_button):
            layout.addWidget(widget)
        self.border_checkbox = QCheckBox("框线")
        self.border_checkbox.setChecked(bool(reading.get("border_visible", False)))
        self.border_checkbox.toggled.connect(self.border_changed.emit)
        self.hide_text_checkbox = QCheckBox("隐藏译文")
        self.hide_text_checkbox.setChecked(bool(reading.get("hide_text", False)))
        self.hide_text_checkbox.toggled.connect(self.hide_text_changed.emit)
        for checkbox in (self.border_checkbox, self.hide_text_checkbox):
            checkbox.setFocusPolicy(Qt.NoFocus)
            layout.addWidget(checkbox)
        self.status_label = QLabel("准备中")
        self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setMinimumWidth(60)
        layout.addWidget(self.status_label, 1)
        self.set_ready(False)
        self.mark_button.setEnabled(False)
        self.peek_button.setEnabled(False)
        self.setFixedHeight(layout.sizeHint().height())

    def set_running(self, running):
        self._running = bool(running)
        self.toggle_button.setText("暂停" if self._running else "开始")
        self.toggle_button.setEnabled(self._running or self._ready)
        # 标记与单句对照只在阅读进行时可用；暂停/退出时自动关闭，避免覆盖层一直抢点击。
        for button in (self.mark_button, self.peek_button):
            button.setEnabled(self._running)
            if not self._running and button.isChecked():
                button.setChecked(False)

    def set_status(self, status):
        status = str(status)
        self.status_label.setToolTip(status)
        self.status_label.setText(self.status_label.fontMetrics().elidedText(
            status, Qt.ElideRight, max(60, self.status_label.width())))

    def set_ready(self, ready):
        self._ready = bool(ready)
        self.toggle_button.setEnabled(self._running or self._ready)
        self.refresh_button.setEnabled(self._ready)

    # ---------- 译文样式 ----------
    def style(self):
        return self._font_scale, self._color, self._auto_color, self._stroke_ratio

    def apply_style(self, scale, color, auto_color=None, stroke_ratio=None):
        """记住样式并通知外部；实际重排在控制器里做。"""
        try:
            self._font_scale = float(scale)
        except (TypeError, ValueError):
            return
        self._color = str(color or self._color)
        if auto_color is not None:
            self._auto_color = bool(auto_color)
        if stroke_ratio is not None:
            try:
                self._stroke_ratio = float(stroke_ratio)
            except (TypeError, ValueError):
                pass
        self.style_changed.emit(self._font_scale, self._color,
                                self._auto_color, self._stroke_ratio)

    def open_style_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("译文样式")
        dialog.setStyleSheet(_DARK_QSS)
        form = QFormLayout(dialog)
        scale_box = QDoubleSpinBox(dialog)
        scale_box.setRange(0.5, 2.0)
        scale_box.setSingleStep(0.1)
        scale_box.setDecimals(1)
        scale_box.setValue(self._font_scale)
        scale_box.setToolTip("大于 1.0 时中文更大，但可能超出原气泡范围")

        auto_box = QCheckBox("按背景自动配色", dialog)
        auto_box.setChecked(self._auto_color)
        auto_box.setToolTip("亮底用深字、暗底用浅字；背景复杂时靠它才看得清")

        stroke_box = QDoubleSpinBox(dialog)
        stroke_box.setRange(0.0, 20.0)
        stroke_box.setSingleStep(1.0)
        stroke_box.setDecimals(0)
        stroke_box.setSuffix(" %")
        stroke_box.setValue(self._stroke_ratio * 100)
        stroke_box.setToolTip("描边宽度占字号的比例，0 表示不描边；描边色自动取字色反色")

        chosen = {"color": self._color}
        color_button = QPushButton(self._color, dialog)
        color_button.setToolTip("固定字色；勾选自动配色后该项不生效")

        def sync_color_enabled(checked):
            color_button.setEnabled(not checked)

        def pick_color():
            picked = QColorDialog.getColor(QColor(chosen["color"]), dialog, "选择译文颜色")
            if picked.isValid():
                chosen["color"] = picked.name()
                color_button.setText(chosen["color"])

        auto_box.toggled.connect(sync_color_enabled)
        sync_color_enabled(auto_box.isChecked())
        color_button.clicked.connect(pick_color)
        form.addRow("字号缩放", scale_box)
        form.addRow(auto_box)
        form.addRow("文字颜色", color_button)
        form.addRow("描边", stroke_box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.Accepted:
            self.apply_style(scale_box.value(), chosen["color"],
                             auto_box.isChecked(), stroke_box.value() / 100)


class NativePanel(QWidget):
    pick_requested = Signal()
    single_requested = Signal(str)
    realtime_requested = Signal(str)
    provider_requested = Signal(str)
    inpaint_requested = Signal(str)
    source_language_requested = Signal(str)
    grain_requested = Signal(bool)
    recognize_requested = Signal(str)
    prompts_requested = Signal(str, str)
    hotkeys_requested = Signal(dict)
    test_connection_requested = Signal()
    model_requested = Signal(str)
    preset_requested = Signal(dict)
    export_requested = Signal()
    exit_requested = Signal()

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._ready = False
        self._busy = False
        self._realtime = False
        self.setWindowTitle("漫画翻译 · 设置与调试")
        self.setStyleSheet(_DARK_QSS)
        self.resize(820, 680)
        layout = QVBoxLayout(self)
        coordinates = QGridLayout()
        self.fields = []
        for column, (name, label) in enumerate((("left", "左"), ("top", "上"), ("width", "宽"), ("height", "高"))):
            coordinates.addWidget(QLabel(label + "（物理像素）"), 0, column)
            field = QLineEdit()
            field.setObjectName("region_" + name)
            coordinates.addWidget(field, 1, column)
            self.fields.append(field)
        layout.addLayout(coordinates)
        self.set_region(",".join(map(str, cfg.get("capture", {}).get("region", [0, 0, 0, 0]))))
        controls = QHBoxLayout()
        self.pick_button = QPushButton("框选")
        self.single_button = QPushButton("翻译单页")
        self.recognize_button = QPushButton("只识别")
        self.recognize_button.setToolTip(
            "只跑 OCR、不抹字不翻译，把逐块原文和坐标标在图上。\n"
            "排查「识别错」还是「翻译错」时用它。")
        self.realtime_button = QPushButton("进入透明实时")
        self.exit_button = QPushButton("退出")
        self.provider_combo = QComboBox()
        for provider, label in (("free", "免费 / 本地"), ("paid", "付费 / 云端"), ("mock", "测试 / 原样回填")):
            self.provider_combo.addItem(label, provider)
        provider = cfg.get("llm", {}).get("provider", "free")
        index = self.provider_combo.findData(provider)
        if index < 0:
            self.provider_combo.addItem(str(provider), provider)
            index = self.provider_combo.count() - 1
        self.provider_combo.setCurrentIndex(index)
        self.provider_combo.currentIndexChanged.connect(
            lambda index: self.provider_requested.emit(str(self.provider_combo.itemData(index))))
        self.inpaint_combo = QComboBox()
        for method, label in (("cover", "抹字：快速"), ("lama", "抹字：LaMa 修复（慢）")):
            self.inpaint_combo.addItem(label, method)
        self.inpaint_combo.setToolTip(
            "快速：纯色覆盖，复杂背景用局部 inpaint；默认不添加随机颗粒。\n"
            "LaMa：深度学习修复，较慢，不能保证还原被文字遮住的网点和线条。")
        method = cfg.get("inpaint", {}).get("method", "cover")
        index = self.inpaint_combo.findData(method)
        if index < 0:
            self.inpaint_combo.addItem(str(method), method)
            index = self.inpaint_combo.count() - 1
        self.inpaint_combo.setCurrentIndex(index)
        self.inpaint_combo.currentIndexChanged.connect(
            lambda index: self.inpaint_requested.emit(str(self.inpaint_combo.itemData(index))))
        self.connect_button = QPushButton("连接/模型")
        self.connect_button.setObjectName("connect_button")
        self.connect_button.setToolTip(
            "测试当前通道连接并列出服务端可用模型（GET /models），顺带验证 key 是否有效。\n"
            "选中模型即写回该通道，本次会话生效；想永久保存请写进 config.yaml。")
        self.connect_button.clicked.connect(lambda: self.test_connection_requested.emit())
        self.pick_button.clicked.connect(lambda: self.pick_requested.emit())
        self.single_button.clicked.connect(lambda: self.single_requested.emit(self._region_text()))
        self.recognize_button.clicked.connect(lambda: self.recognize_requested.emit(self._region_text()))
        # 不自行改变实时状态；暂停也将当前 region 交给主调度执行 toggle。
        self.realtime_button.clicked.connect(lambda: self.realtime_requested.emit(self._region_text()))
        self.exit_button.clicked.connect(lambda: self.exit_requested.emit())
        for widget in (self.pick_button, self.single_button, self.recognize_button,
                       self.realtime_button, self.provider_combo, self.inpaint_combo,
                       self.connect_button, self.exit_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        quality_controls = QHBoxLayout()
        self.presets_combo = QComboBox()
        self.presets_combo.setObjectName("presets_combo")
        self.presets_combo.addItem("预设：选择即填入", None)
        self._presets = {}
        for preset in cfg.get("llm", {}).get("presets") or []:
            if not isinstance(preset, dict):
                continue
            name = str(preset.get("name") or "").strip()
            if not name or name in self._presets:
                continue
            self._presets[name] = preset
            self.presets_combo.addItem(name, name)
        self.presets_combo.setToolTip(
            "一键填入供应商的地址、模型与请求参数（会整体替换 llm.options）。\n"
            "只对本次会话生效；想永久保存就把选中的预设写进 config.yaml。")
        self.presets_combo.activated.connect(self._apply_preset)
        quality_controls.addWidget(self.presets_combo)
        quality_controls.addWidget(QLabel("原文语言"))
        self.source_language_combo = QComboBox()
        for language, label in (("auto", "自动（混合界面）"), ("ja", "日文（日漫推荐）"),
                                ("en", "英文"), ("ko", "韩文"), ("ru", "俄文")):
            self.source_language_combo.addItem(label, language)
        language = cfg.get("llm", {}).get("source_lang", "auto")
        self.source_language_combo.setCurrentIndex(max(0, self.source_language_combo.findData(language)))
        self.source_language_combo.setToolTip(
            "日漫建议固定日文：纯汉字日文/人名在自动模式下可能被当成中文跳过。\n"
            "固定语言时请只框漫画，不要包含聊天窗或浏览器界面文字。")
        self.source_language_combo.currentIndexChanged.connect(
            lambda index: self.source_language_requested.emit(str(self.source_language_combo.itemData(index))))
        self.grain_check = QCheckBox("随机颗粒（实验，默认关闭）")
        self.grain_check.setChecked(bool(cfg.get("inpaint", {}).get("grain", False)))
        self.grain_check.setToolTip("随机噪声不等于漫画网点；此开关只影响快速抹字。设置只对本次会话生效。")
        self.grain_check.toggled.connect(self.grain_requested.emit)
        self.prompt_button = QPushButton("提示词")
        self.prompt_button.setObjectName("prompt_button")
        self.prompt_button.setToolTip(
            "编辑翻译提示词，支持 $from / $to / $detect / $text 占位符。\n"
            "留空则用内置默认；修改立即生效，不影响下次启动的 config.yaml。")
        self.prompt_button.clicked.connect(self.open_prompt_dialog)
        self.hotkey_button = QPushButton("快捷键")
        self.hotkey_button.setObjectName("hotkey_button")
        self.hotkey_button.setToolTip(
            "设置全局快捷键，阅读时不用切窗口就能开始/暂停、重选、刷新。\n"
            "仅 Windows 生效；组合被别的程序占用时那一个会失效并提示，其余照常。")
        self.hotkey_button.clicked.connect(self.open_hotkey_dialog)
        self.export_button = QPushButton("导出译文")
        self.export_button.setObjectName("export_button")
        self.export_button.setToolTip(
            "把译文按「原句 / 译句」两行一组导出成文本，方便扫一遍校对。\n"
            "实时阅读会累积翻过的每一页；导出位置见运行日志。")
        self.export_button.clicked.connect(self.export_requested.emit)
        quality_controls.addWidget(self.source_language_combo)
        quality_controls.addWidget(self.grain_check)
        quality_controls.addWidget(self.prompt_button)
        quality_controls.addWidget(self.hotkey_button)
        quality_controls.addWidget(self.export_button)
        quality_controls.addStretch()
        layout.addLayout(quality_controls)
        layout.addWidget(QLabel("回填图预览（完整图仅在本调试面板显示）"))
        self.preview_label = QLabel("尚无结果")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setWidgetResizable(True)
        layout.addWidget(self.preview_scroll, 3)
        layout.addWidget(QLabel("原文 / 译文"))
        self.pairs_view = QPlainTextEdit()
        self.pairs_view.setReadOnly(True)
        layout.addWidget(self.pairs_view, 2)
        layout.addWidget(QLabel("运行日志"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        layout.addWidget(self.log_view, 1)
        self._update_controls()

    def _region_text(self):
        return ",".join(field.text().strip() for field in self.fields)

    def _apply_preset(self, index):
        """套用预设：交给主调度写配置，再同步通道下拉，标记复位以便重复选择。"""
        preset = self._presets.get(str(self.presets_combo.itemData(index)))
        self.presets_combo.setCurrentIndex(0)
        if preset is None:
            return
        self.preset_requested.emit(preset)
        provider_index = self.provider_combo.findData(str(preset.get("provider") or ""))
        if provider_index >= 0 and provider_index != self.provider_combo.currentIndex():
            self.provider_combo.setCurrentIndex(provider_index)

    def set_region(self, region):
        parts = str(region).strip().strip("[]()").replace("，", ",").split(",")
        if len(parts) != 4:
            raise ValueError("区域格式应为 左,上,宽,高")
        for field, value in zip(self.fields, parts):
            field.setText(value.strip())

    def _update_controls(self):
        editable = not self._busy and not self._realtime
        for field in self.fields:
            field.setEnabled(editable)
        self.pick_button.setEnabled(editable)
        self.provider_combo.setEnabled(editable)
        self.presets_combo.setEnabled(editable)
        self.inpaint_combo.setEnabled(editable)
        self.source_language_combo.setEnabled(editable)
        self.grain_check.setEnabled(editable)
        self.prompt_button.setEnabled(editable)
        self.hotkey_button.setEnabled(editable)
        self.connect_button.setEnabled(editable)
        self.export_button.setEnabled(editable)
        self.single_button.setEnabled(self._ready and editable)
        self.recognize_button.setEnabled(self._ready and editable)
        self.realtime_button.setEnabled(self._realtime or (self._ready and not self._busy))
        self.realtime_button.setText("暂停实时" if self._realtime else "进入透明实时")

    def set_busy(self, busy):
        self._busy = bool(busy)
        self._update_controls()

    def set_realtime(self, realtime):
        self._realtime = bool(realtime)
        self._update_controls()

    def set_ready(self, ready):
        self._ready = bool(ready)
        self._update_controls()

    def append_log(self, message):
        self.log_view.appendPlainText(str(message))

    def open_prompt_dialog(self):
        """编辑翻译提示词；留空表示用内置默认。"""
        from engine.translate import DEFAULT_BATCH_PROMPT, DEFAULT_SYSTEM_PROMPT
        prompt_cfg = self._cfg.get("llm", {}).get("prompts", {}) or {}
        dialog = QDialog(self)
        dialog.setWindowTitle("翻译提示词")
        dialog.setStyleSheet(_DARK_QSS)
        dialog.resize(640, 460)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("可用占位符：$from 原文语言　$to 目标语言　$detect 识别语言　$text 原文"))
        layout.addWidget(QLabel("系统提示词（留空用默认）"))
        system_edit = QPlainTextEdit(dialog)
        system_edit.setObjectName("prompt_system")
        system_edit.setPlainText(str(prompt_cfg.get("system") or ""))
        system_edit.setPlaceholderText(DEFAULT_SYSTEM_PROMPT)
        layout.addWidget(system_edit, 1)
        layout.addWidget(QLabel("批量翻译提示词（留空用默认；$text 是带序号的原文，必须保留）"))
        batch_edit = QPlainTextEdit(dialog)
        batch_edit.setObjectName("prompt_batch")
        batch_edit.setPlainText(str(prompt_cfg.get("batch") or ""))
        batch_edit.setPlaceholderText(DEFAULT_BATCH_PROMPT)
        layout.addWidget(batch_edit, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        reset_button = buttons.addButton("恢复默认", QDialogButtonBox.ResetRole)

        def reset():
            system_edit.clear()
            batch_edit.clear()

        reset_button.clicked.connect(reset)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.Accepted:
            self.prompts_requested.emit(system_edit.toPlainText().strip(),
                                        batch_edit.toPlainText().strip())

    def open_hotkey_dialog(self):
        """设置全局快捷键；每个动作一个输入框，留空表示不注册。"""
        from engine.hotkey import (HotkeyError, format_hotkey, hotkey_available,
                                   parse_hotkey)
        labels = {
            "toggle": "开始 / 暂停实时",
            "reselect": "重新框选区域",
            "refresh": "刷新当前页",
            "settings": "回到设置页",
            "leave": "退出透明阅读",
        }
        current = dict(self._cfg.get("hotkeys", {}) or {})
        dialog = QDialog(self)
        dialog.setWindowTitle("全局快捷键")
        dialog.setStyleSheet(_DARK_QSS)
        dialog.resize(520, 360)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("仅 Windows 生效。写法：修饰键 + 一个主键，例如 ctrl+alt+s\n"
                                "留空表示不注册该动作；点「确定」后立即生效，并写回 config.yaml。"))
        form = QFormLayout()
        editors = {}
        for action, label in labels.items():
            editor = QLineEdit(dialog)
            editor.setObjectName("hotkey_" + action)
            editor.setPlaceholderText("例如 ctrl+alt+s（留空=不注册）")
            editor.setText(str(current.get(action) or ""))
            editors[action] = editor
            form.addRow(label, editor)
        layout.addLayout(form)
        hint = QLabel("")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)

        def clear_all():
            for editor in editors.values():
                editor.clear()
            hint.setText("已清空，点「确定」后所有快捷键停用")

        buttons.addButton("全部清空", QDialogButtonBox.ResetRole).clicked.connect(clear_all)

        def accept():
            chosen = {}
            problems = []
            seen = {}
            for action, editor in editors.items():
                raw = editor.text().strip()
                if not raw:
                    chosen[action] = ""
                    continue
                try:
                    normalized = format_hotkey(raw)
                    parse_hotkey(raw)
                except HotkeyError as error:
                    problems.append(f"{labels[action]}：{error}")
                    continue
                # 同一个键绑两个动作没意义，后者会把前者顶掉，给出明确提示。
                key = normalized.lower()
                if key in seen:
                    problems.append(f"{labels[action]}：{normalized} 和「{labels[seen[key]]}」重复了")
                    continue
                seen[key] = action
                chosen[action] = raw
            if problems:
                hint.setText("请先修正：\n" + "\n".join(problems))
                return
            # 占用检测：注册失败不影响其他键，但要让用户先知道。
            occupied = []
            for action, raw in chosen.items():
                if not raw:
                    continue
                ok, why = hotkey_available(raw)
                if not ok:
                    occupied.append(f"{labels[action]}：{why}")
            if occupied:
                hint.setText("以下组合可能不可用（仍会保存，启动时如失败会在日志提示）：\n"
                             + "\n".join(occupied))
                return
            dialog.accept()
            self.hotkeys_requested.emit(chosen)

        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def show_models_result(self, payload):
        """连接测试结果：成功时列出模型供选择，失败给出具体原因。"""
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            data = {}
        if not isinstance(data, dict) or not data.get("ok"):
            error = "结果格式错误" if not isinstance(data, dict) else str(data.get("error") or "未知错误")
            self.append_log("连接测试失败：" + error)
            QMessageBox.warning(self, "连接测试", error)
            return
        provider = str(data.get("provider", ""))
        models = [str(m) for m in data.get("models", []) if str(m).strip()]
        if not models:
            message = "连接成功，但接口没有返回模型列表；可在 config.yaml 里直接填写模型名。"
            self.append_log(message)
            QMessageBox.information(self, "连接测试", message)
            return
        current = str(data.get("current") or "")
        index = models.index(current) if current in models else 0
        choice, accepted = QInputDialog.getItem(
            self, "选择模型", f"{provider} 通道可用模型（写入本次会话）：", models, index, False)
        if accepted and choice:
            self.model_requested.emit(choice)

    def show_result(self, result):
        self.preview_label.clear()
        self.preview_label.setMinimumSize(0, 0)
        self.pairs_view.clear()
        try:
            payload = json.loads(result)
            if not isinstance(payload, dict):
                raise ValueError("结果必须是 JSON 对象")
            if payload.get("ok") is False:
                self.append_log("出错：" + str(payload.get("error", "未知错误")))
                return
            image = _decode_image(payload.get("image"))
            if not image.isNull():
                self.preview_label.setPixmap(QPixmap.fromImage(image))
                self.preview_label.setMinimumSize(image.size())
            translations = payload.get("translations", {})
            if payload.get("recognize_only"):
                # 诊断模式：逐块列编号 / 位置 / 横竖排 / 置信度 / 原文，便于对照标注图。
                lines = []
                for i, box in enumerate(payload.get("boxes", []), 1):
                    if not isinstance(box, dict):
                        continue
                    tag = "竖排" if box.get("vertical") else "横排"
                    lines.append(f'[{i}] {tag} ({box.get("x")},{box.get("y")},'
                                 f'{box.get("w")},{box.get("h")}) 置信度 {box.get("score", 0):.2f}\n'
                                 f'    {box.get("text", "")}')
                language = payload.get("language", "unknown")
                self.pairs_view.setPlainText(
                    f"判定原文语言：{language}（{len(payload.get('boxes', []))} 个文字块）\n\n"
                    + "\n\n".join(lines))
                return
            pairs = payload.get("pairs", translations)
            lines = []
            if isinstance(pairs, dict):
                lines = [f"{source}\n{target}" for source, target in pairs.items()]
            elif isinstance(pairs, list):
                for pair in pairs:
                    if isinstance(pair, dict):
                        lines.append(f"{pair.get('source', pair.get('src', ''))}\n{pair.get('translation', pair.get('target', pair.get('dst', '')))}")
                    elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                        lines.append(f"{pair[0]}\n{pair[1]}")
            self.pairs_view.setPlainText("\n\n".join(lines))
        except (TypeError, ValueError) as exc:
            self.append_log("结果读取失败：" + str(exc))

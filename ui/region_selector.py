# -*- coding: utf-8 -*-
"""冻结当前屏幕直接拖拽截图，输出物理像素 [左, 上, 宽, 高]。"""
from PySide6.QtCore import QEventLoop, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RegionSelector(QWidget):
    region_selected = Signal(list)
    cancelled = Signal()

    def __init__(self, screen=None, shot=None):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self._screen = screen or QGuiApplication.primaryScreen()
        if self._screen is None:
            raise RuntimeError("未找到可截图的显示器")
        self.setGeometry(self._screen.geometry())
        self._shot = shot if shot is not None else self._screen.grabWindow(0).toImage()
        if self._shot.isNull():
            raise RuntimeError("屏幕截图失败")
        self._origin = QPoint()
        self._current = QPoint()
        self._selecting = False
        self._finished = False
        self._selected_rect = QRect()

    def _rect(self):
        """半开区间，避免 QRect(QPoint, QPoint) 的包含端点 +1 问题。"""
        x0, x1 = sorted((self._origin.x(), self._current.x()))
        y0, y1 = sorted((self._origin.y(), self._current.y()))
        return QRect(x0, y0, x1 - x0, y1 - y0).intersected(self.rect())

    def _physical_rect(self, rect):
        # 使用实际截图尺寸/窗口尺寸，而非猜测图像和窗口 DPR 是否一致。
        sx = self._shot.width() / self.width()
        sy = self._shot.height() / self.height()
        x0, y0 = round(rect.x() * sx), round(rect.y() * sy)
        x1 = round((rect.x() + rect.width()) * sx)
        y1 = round((rect.y() + rect.height()) * sy)
        return QRect(x0, y0, x1 - x0, y1 - y0).intersected(self._shot.rect())

    def _region(self):
        rect = self._physical_rect(self._rect())
        # Windows Qt 屏幕原点保留原生坐标；只缩放屏幕内局部偏移，不能把整个全局坐标乘 DPR。
        origin = self._screen.geometry().topLeft()
        return [origin.x() + rect.x(), origin.y() + rect.y(), rect.width(), rect.height()]

    def selected_image(self):
        """取冻结背景中的选区，不在蒙层关闭/主窗口恢复后重新抓屏。"""
        return self._shot.copy(self._selected_rect)

    def paintEvent(self, event):
        p = QPainter(self)
        # 显式把完整物理截图映射到完整逻辑窗口，不做隐式尺寸假设。
        p.drawImage(self.rect(), self._shot, self._shot.rect())
        if not self._selecting:
            p.fillRect(self.rect(), QColor(0, 0, 0, 100))
            label = "拖动选择屏幕内容 · 松开完成 · Esc / 右键取消"
            box = QRect(16, 16, p.fontMetrics().horizontalAdvance(label) + 24, 32)
            p.fillRect(box, QColor(25, 25, 30, 230))
            p.setPen(QColor(255, 255, 255))
            p.drawText(box, Qt.AlignCenter, label)
            return
        r = self._rect()
        # 只压暗选框外，选框内不重新采样，从根本上避免内容被放大/错位。
        dark = QColor(0, 0, 0, 100)
        right, bottom = r.x() + r.width(), r.y() + r.height()
        for outside in (QRect(0, 0, self.width(), r.y()),
                        QRect(0, bottom, self.width(), self.height() - bottom),
                        QRect(0, r.y(), r.x(), r.height()),
                        QRect(right, r.y(), self.width() - right, r.height())):
            p.fillRect(outside, dark)
        p.setPen(QPen(QColor(255, 90, 90), 2))
        p.drawRect(r.adjusted(0, 0, -1, -1))
        physical = self._physical_rect(r)
        label = f"{physical.width()} × {physical.height()}"
        width = p.fontMetrics().horizontalAdvance(label) + 16
        x = min(r.x(), max(0, self.width() - width))
        y = max(0, r.y() - 28)
        box = QRect(x, y, width, 24)
        p.fillRect(box, QColor(25, 25, 30, 230))
        p.setPen(QColor(255, 255, 255))
        p.drawText(box, Qt.AlignCenter, label)

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._cancel()
        elif e.button() == Qt.LeftButton and not self._finished:
            self._origin = e.position().toPoint()
            self._current = self._origin
            self._selecting = True
            self.update()

    def mouseMoveEvent(self, e):
        if self._selecting:
            self._current = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._selecting and not self._finished:
            self._current = e.position().toPoint()  # 不依赖最后一条 mouseMove，避免丢失终点
            rect = self._rect()
            self._selecting = False
            if rect.width() < 3 or rect.height() < 3:
                self._cancel()
                return
            self._selected_rect = self._physical_rect(rect)
            self._finished = True
            self.region_selected.emit(self._region())
            self.close()

    def _cancel(self):
        if not self._finished:
            self._finished = True
            self._selecting = False
            self.cancelled.emit()
        self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._cancel()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, event):
        if not self._finished:
            self._finished = True
            self.cancelled.emit()
        event.accept()


def pick_region_modal():
    """保留兼容入口；GUI 桥使用异步选择，不使用本函数。"""
    selector = RegionSelector()
    loop = QEventLoop()
    result = []
    selector.region_selected.connect(lambda r: (result.append(r), loop.quit()))
    selector.cancelled.connect(loop.quit)
    selector.show()
    selector.raise_()
    selector.activateWindow()
    try:
        loop.exec()
    finally:
        selector.close()
    return result[0] if result else None

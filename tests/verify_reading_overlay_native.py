# -*- coding: utf-8 -*-
"""在真实 Windows 桌面上核对透明阅读层的关键机制，不显示任何窗口。

验证三件事：
1. 截图排除：SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) 是否被系统接受；
2. 鼠标穿透：原生扩展样式是否含 WS_EX_TRANSPARENT / WS_EX_LAYERED / WS_EX_NOACTIVATE；
3. 透明背景：窗口是否带 WA_TranslucentBackground 与分层窗口样式。

用法：python tests/verify_reading_overlay_native.py
"""
import ctypes
import os
import sys
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
WDA_EXCLUDEFROMCAPTURE = 0x11


def main():
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication
    from ui.reading import ReadingOverlay, ReadingToolbar

    app = QApplication.instance() or QApplication([])
    platform = QGuiApplication.platformName()
    print("平台:", platform)
    if platform.lower() != "windows":
        print("结果：非 Windows 平台，无法核对截图排除与鼠标穿透，跳过")
        return 2

    cfg = {"reading": {"border_visible": False, "hide_text": False}, "inpaint": {"pad": 3}}
    overlay, toolbar = ReadingOverlay(cfg), ReadingToolbar(cfg)
    # winId() 触发原生窗口创建；不调用 show()，屏幕上不会出现任何窗口。
    overlay.winId()
    toolbar.winId()
    app.processEvents()

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_long = user32.GetWindowLongW
    get_long.argtypes = [wintypes.HWND, ctypes.c_int]
    get_long.restype = ctypes.c_long
    get_affinity = user32.GetWindowDisplayAffinity
    get_affinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    get_affinity.restype = wintypes.BOOL

    failures = []
    # 阅读层必须穿透鼠标；工具栏必须能接收点击，两者期望相反。
    for name, widget, want_passthrough in (("透明阅读层", overlay, True), ("小工具栏", toolbar, False)):
        handle = wintypes.HWND(int(widget.winId()))
        exstyle = get_long(handle, GWL_EXSTYLE)
        affinity = wintypes.DWORD()
        affinity_ok = bool(get_affinity(handle, ctypes.byref(affinity)))
        passthrough = bool(exstyle & WS_EX_TRANSPARENT) and bool(exstyle & WS_EX_LAYERED)
        flags = {
            "截图排除": widget.capture_excluded and affinity_ok and affinity.value == WDA_EXCLUDEFROMCAPTURE,
            "鼠标穿透" if want_passthrough else "可接收点击": passthrough == want_passthrough,
            "不抢焦点": bool(exstyle & WS_EX_NOACTIVATE),
        }
        print(f"\n[{name}] 扩展样式 = 0x{exstyle & 0xFFFFFFFF:08X}，显示亲和性 = "
              f"{affinity.value if affinity_ok else '读取失败'}")
        for label, ok in flags.items():
            print(f"  {'通过' if ok else '未通过'}：{label}")
            if not ok:
                failures.append(f"{name}/{label}")

    transparent = overlay.testAttribute(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.WA_TranslucentBackground)
    print("\n[透明背景] WA_TranslucentBackground =", transparent)
    if not transparent:
        failures.append("透明阅读层/透明背景")

    overlay.close()
    toolbar.close()
    print("\n结论：" + ("全部通过，可在真机上做阅读验收" if not failures
                      else "存在未通过项：" + "、".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

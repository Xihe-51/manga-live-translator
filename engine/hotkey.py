# -*- coding: utf-8 -*-
"""全局快捷键（Windows RegisterHotKey）。

为什么要自己写而不是用 keyboard/pynput：本项目已有 pywin32，且 RegisterHotKey
是系统级注册——冲突会被系统明确拒绝（返回 False），比装钩子轮询更省电、更可靠，
失败也能如实告诉用户「这个组合被别的程序占了」。

线程模型：Windows 的消息队列是「线程私有」的，RegisterHotKey 注册出来的热键
只会投递到**注册它的那个线程**的消息队列。所以这里开一条独立线程跑消息泵，
收到 WM_HOTKEY 后用 Qt 信号（队列连接）安全地回到主线程执行 GUI 操作。
"""
from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

# ---- Win32 常量 ----
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000          # 按住不放时不重复触发
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

#: 修饰键名 → 标志位。容忍常见别名，免得用户写 ctrl/control 一个行一个不行。
_MODIFIERS = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
    "super": MOD_WIN,
    "cmd": MOD_WIN,
}

#: 主键名 → 虚拟键码。只收常用键，避免把整个 VK 表抄进来。
_VK_NAMES = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "esc": 0x1B, "escape": 0x1B, "space": 0x20, "spacebar": 0x20,
    "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "insert": 0x2D, "delete": 0x2E, "del": 0x2E,
    "`": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD,
    "\\": 0xDC, ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
}
for _i in range(1, 25):        # F1..F24
    _VK_NAMES[f"f{_i}"] = 0x6F + _i
for _c in "0123456789":        # 0-9
    _VK_NAMES[_c] = 0x30 + int(_c)
for _c in "abcdefghijklmnopqrstuvwxyz":   # A-Z
    _VK_NAMES[_c] = ord(_c.upper())

#: 反向表，用于把「已注册的组合」显示回给用户。
_VK_TO_NAME = {}
for _name, _vk in _VK_NAMES.items():
    _VK_TO_NAME.setdefault(_vk, _name)

_NAME_TO_VK_LABEL = {
    "backspace": "Backspace", "tab": "Tab", "enter": "Enter", "esc": "Esc",
    "space": "Space", "pageup": "PageUp", "pagedown": "PageDown", "end": "End",
    "home": "Home", "left": "Left", "up": "Up", "right": "Right", "down": "Down",
    "insert": "Insert", "delete": "Delete", "`": "`", "-": "-", "=": "=",
    "[": "[", "]": "]", "\\": "\\", ";": ";", "'": "'", ",": ",", ".": ".", "/": "/",
}


class HotkeyError(ValueError):
    """快捷键字符串不合法。"""


def parse_hotkey(text):
    """把 "ctrl+alt+t" 解析成 (modifiers, vk)。

    规则：
      - 至少要有 1 个修饰键，且必须有且仅有 1 个主键；
      - 只有修饰键没主键（"ctrl+alt"）直接报错——那种组合没有意义还容易误注册；
      - 大小写、空格、全角加号都容忍，用户手写不用太较真。
    """
    if text is None:
        raise HotkeyError("快捷键为空")
    raw = str(text).strip()
    if not raw:
        raise HotkeyError("快捷键为空")
    # 全角加号/全角空格统一成半角，顺手把中文用户常打错的形式吃掉。
    raw = raw.replace("＋", "+").replace("\u3000", " ")
    parts = [p.strip().lower() for p in raw.split("+")]
    parts = [p for p in parts if p]
    if not parts:
        raise HotkeyError("快捷键为空")

    modifiers = 0
    keys = []
    for part in parts:
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
        elif part in _VK_NAMES:
            keys.append(part)
        else:
            raise HotkeyError(f"无法识别的按键：{part}")

    if not modifiers:
        raise HotkeyError("至少要有一个修饰键（Ctrl / Alt / Shift / Win）")
    if not keys:
        raise HotkeyError("缺少主键，例如 ctrl+alt+t 里的 t")
    if len(keys) > 1:
        raise HotkeyError("只能有一个主键：" + "+".join(keys))
    return modifiers, _VK_NAMES[keys[0]]


def format_hotkey(text):
    """把用户输入的组合规范成统一显示形式，例如 "CTRL+ALT+T"。"""
    modifiers, vk = parse_hotkey(text)
    order = ((MOD_CONTROL, "Ctrl"), (MOD_ALT, "Alt"), (MOD_SHIFT, "Shift"), (MOD_WIN, "Win"))
    labels = [label for flag, label in order if modifiers & flag]
    name = _VK_TO_NAME.get(vk, str(vk))
    labels.append(_NAME_TO_VK_LABEL.get(name, name.upper()))
    return "+".join(labels)


def hotkey_available(text):
    """探测某个组合当前是否可注册。

    临时注册再立刻注销：能注册就说没被占。探测本身会产生一次系统级占用，
    所以只在用户点「确定」时用，不要放进刷新的循环里。
    """
    try:
        modifiers, vk = parse_hotkey(text)
    except HotkeyError:
        return False, "格式不对"
    if sys.platform != "win32":
        return False, "当前系统不是 Windows"
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    probe_id = 0xBFFF   # 探测专用 id，和正式注册的 id 段隔开
    if not user32.RegisterHotKey(None, probe_id, modifiers | MOD_NOREPEAT, vk):
        return False, f"这个组合已被其他程序占用（错误码 {ctypes.get_last_error()}）"
    user32.UnregisterHotKey(None, probe_id)
    return True, ""


def parse_poll_key(text):
    """解析「按住检测」用键：允许不带修饰键（如 tab），也允许 ctrl+tab。

    与 parse_hotkey 的差别只在修饰键可省略——按住对照是瞬时动作，
    用户多半直接按 Tab/F8 这类单键，强行要求修饰键就没人用得了。
    """
    if text is None:
        raise HotkeyError("未配置按键")
    raw = str(text).strip()
    if not raw:
        raise HotkeyError("未配置按键")
    raw = raw.replace("＋", "+").replace("\u3000", " ")
    parts = [p.strip().lower() for p in raw.split("+")]
    parts = [p for p in parts if p]
    modifiers, keys = 0, []
    for part in parts:
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
        elif part in _VK_NAMES:
            keys.append(part)
        else:
            raise HotkeyError(f"无法识别的按键：{part}")
    if len(keys) != 1:
        raise HotkeyError("按住键只能是一个按键，例如 tab 或 ctrl+tab")
    return modifiers, _VK_NAMES[keys[0]]


def format_poll_key(text):
    """把按住键规范成显示形式，例如 "Tab"、"Ctrl+Tab"。"""
    modifiers, vk = parse_poll_key(text)
    order = ((MOD_CONTROL, "Ctrl"), (MOD_ALT, "Alt"), (MOD_SHIFT, "Shift"), (MOD_WIN, "Win"))
    labels = [label for flag, label in order if modifiers & flag]
    name = _VK_TO_NAME.get(vk, str(vk))
    labels.append(_NAME_TO_VK_LABEL.get(name, name.upper()))
    return "+".join(labels)


class HotkeyThread(QThread):
    """独立线程跑 Windows 消息泵，把 WM_HOTKEY 转成 Qt 信号。"""

    triggered = Signal(str)      # 发出动作名（action）
    failed = Signal(str, str)    # (action, 错误说明)
    ready = Signal(dict)         # 注册阶段结束，发出 {action: "Ctrl+Alt+T"}

    def __init__(self, bindings=None, parent=None):
        """bindings: {action: "ctrl+alt+t"}；值为空表示不注册该动作。"""
        super().__init__(parent)
        self._bindings = dict(bindings or {})
        self._id_to_action = {}
        self._registered = {}        # action -> (hotkey 文本, id)
        self._failures = {}          # action -> 错误说明
        self._thread_id = None
        # 注册阶段的结果由工作线程写入、主线程读取，用锁保证读到的不是半成品。
        self._lock = threading.Lock()
        self._done = threading.Event()

    @property
    def registered(self):
        with self._lock:
            return {action: text for action, (text, _) in self._registered.items()}

    @property
    def failures(self):
        with self._lock:
            return dict(self._failures)

    def wait_until_registered(self, timeout=2.0):
        """阻塞等待注册阶段结束。

        只在需要「同步知道注册结果」的场合调用（例如测试）；消息泵启动后
        本线程会一直阻塞在 GetMessageW，所以不能用 QThread.wait() 等它退出。
        """
        return self._done.wait(timeout)

    def run(self):
        if sys.platform != "win32":
            for action in self._bindings:
                self._record_failure(action, "当前系统不是 Windows，全局快捷键不可用")
            self._done.set()
            self.ready.emit({})
            return

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int,
                                          wintypes.UINT, wintypes.UINT]
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.UnregisterHotKey.restype = wintypes.BOOL
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                       wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = ctypes.c_int

        self._thread_id = kernel32.GetCurrentThreadId()

        # 先全部注册。用递增 id，方便报错时定位是哪个动作。
        for index, (action, text) in enumerate(self._bindings.items(), 1):
            if not text:
                continue
            try:
                modifiers, vk = parse_hotkey(text)
            except HotkeyError as error:
                self._record_failure(action, str(error))
                continue
            if user32.RegisterHotKey(None, index, modifiers | MOD_NOREPEAT, vk):
                with self._lock:
                    self._id_to_action[index] = action
                    self._registered[action] = (format_hotkey(text), index)
            else:
                self._record_failure(
                    action,
                    f"{format_hotkey(text)} 注册失败，可能已被其他程序占用"
                    f"（错误码 {ctypes.get_last_error()}）。"
                    f"若确认没有别的软件用这个组合，多半是本程序上一次的实例"
                    f"还没退干净——在任务管理器里结束残留的 python 进程再启动。")

        # 注册阶段结束就通知外部拿结果。不能用 QThread.wait() 等——消息泵会
        # 一直阻塞在 GetMessageW，wait() 只有等到线程退出才返回，主线程会被卡住。
        self._done.set()
        self.ready.emit(self.registered)

        message = wintypes.MSG()
        while True:
            got = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
            if got <= 0:          # 0 = WM_QUIT，-1 = 出错
                break
            if message.message == WM_HOTKEY:
                action = self._id_to_action.get(int(message.wParam))
                if action:
                    self.triggered.emit(action)
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))

        for index in list(self._id_to_action):
            user32.UnregisterHotKey(None, index)
        self._id_to_action.clear()

    def _record_failure(self, action, error):
        with self._lock:
            self._failures[action] = error
        self.failed.emit(action, error)

    def stop(self):
        """请求线程退出：往它的消息队列投 WM_QUIT 打断 GetMessage。"""
        if sys.platform != "win32" or self._thread_id is None:
            return
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.PostThreadMessageW(wintypes.DWORD(self._thread_id), WM_QUIT, 0, 0)
        except (AttributeError, OSError):
            pass


class HotkeyManager(QObject):
    """对外统一入口：装载配置、转发动作、出错降级但不影响主程序。"""

    triggered = Signal(str)
    message = Signal(str)          # 给人看的日志/提示

    def __init__(self, bindings=None, parent=None):
        super().__init__(parent)
        self._bindings = dict(bindings or {})
        self._thread = None
        self._registered = {}
        self._failures = {}

    @property
    def registered(self):
        return dict(self._registered)

    @property
    def failures(self):
        """{action: 错误说明}，让 UI 能明确告诉用户哪个组合没生效。"""
        return dict(self._failures)

    @property
    def enabled(self):
        return self._thread is not None and self._thread.isRunning()

    def start(self):
        """按当前 bindings 注册热键。已注册的组合冲突时只提示、不中断启动。"""
        self.stop()
        self._failures = {}
        if not any(self._bindings.values()):
            return
        thread = HotkeyThread(self._bindings, parent=self)
        thread.triggered.connect(self.triggered)
        thread.failed.connect(self._on_failed)
        thread.ready.connect(self._on_ready)
        thread.start()
        self._thread = thread
        # 等注册阶段收尾，这样调用方 start() 一返回就能读到 registered/failures，
        # 不依赖主线程事件循环什么时候转起来（启动瞬间还可能是空的）。
        thread.wait_until_registered(2.0)
        self._registered = thread.registered
        self._failures = thread.failures
        if self._registered:
            summary = "、".join(f"{a}:{k}" for a, k in sorted(self._registered.items()))
            self.message.emit("全局快捷键已生效（" + summary + "）")
        for action, error in sorted(self._failures.items()):
            self.message.emit(f"快捷键 {action} 不可用：{error}")

    @Slot(dict)
    def _on_ready(self, registered):
        # 线程侧的异步通知：结果已由 start() 同步取过，这里只做兜底，避免漏报。
        if not self._registered and registered:
            self._registered = dict(registered)

    def _on_failed(self, action, error):
        # 单个组合注册失败不能影响其他键，也不能让程序起不来。
        self._failures[action] = error

    def stop(self):
        thread, self._thread = self._thread, None
        self._registered = {}
        if thread is None:
            return
        thread.stop()
        thread.wait(1000)
        thread.deleteLater()

    def rebind(self, bindings):
        """改绑：先全释放再重新注册，避免旧组合残留占用。"""
        self._bindings = dict(bindings or {})
        self.start()


#: 修饰键标志 → 需要检查的物理键码（Win 键左右各一个）。
_MODIFIER_KEYCODES = (
    (MOD_CONTROL, (0x11,)),      # VK_CONTROL
    (MOD_ALT, (0x12,)),          # VK_MENU
    (MOD_SHIFT, (0x10,)),        # VK_SHIFT
    (MOD_WIN, (0x5B, 0x5C)),     # VK_LWIN / VK_RWIN
)


class HoldKeyMonitor(QObject):
    """轮询物理键状态，检测「按住 / 松开」。

    RegisterHotKey 只投递按下事件、拿不到松开，做不了"按住看原图"这类瞬时对照，
    只能用 QTimer 轮询 GetAsyncKeyState；阅读期间才启动，开销可忽略。
    非 Windows 或按键不合法时静默不可用（available=False），不影响主程序。
    """

    pressed = Signal()
    released = Signal()

    def __init__(self, key_text, interval_ms=60, parent=None):
        super().__init__(parent)
        self._modifiers, self._vk = 0, None
        self._label = ""
        self._user32 = None
        self._down = False
        try:
            self._modifiers, self._vk = parse_poll_key(key_text)
            self._label = format_poll_key(key_text)
        except HotkeyError:
            self._modifiers, self._vk = 0, None
        if self._vk is not None and sys.platform == "win32":
            try:
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
                user32.GetAsyncKeyState.restype = wintypes.SHORT
                self._user32 = user32
            except (AttributeError, OSError):
                self._user32 = None
        self.timer = QTimer(self)
        self.timer.setInterval(max(30, int(interval_ms)))
        self.timer.timeout.connect(self._poll)

    @property
    def available(self):
        return self._user32 is not None

    @property
    def label(self):
        """给界面显示用的规范按键名；未配置/不合法时为空串。"""
        return self._label

    def start(self):
        if self.available:
            self.timer.start()

    def stop(self):
        self.timer.stop()
        if self._down:
            # 停的时候还按着：补发一次「松开」，否则界面会停在按住的样子。
            self._down = False
            self.released.emit()

    def _poll(self):
        if not self.available:
            return
        down = self._is_pressed()
        if down == self._down:
            return
        self._down = down
        (self.pressed if down else self.released).emit()

    def _is_pressed(self):
        for flag, keycodes in _MODIFIER_KEYCODES:
            if self._modifiers & flag and not any(
                    self._user32.GetAsyncKeyState(code) & 0x8000 for code in keycodes):
                return False
        return bool(self._user32.GetAsyncKeyState(self._vk) & 0x8000)


def hotkeys_from_config(cfg):
    """从 config.yaml 的 hotkeys 段取绑定；缺项/空串表示不注册。"""
    section = (cfg or {}).get("hotkeys", {}) or {}
    return {str(action): (str(text).strip() if text else "")
            for action, text in section.items()}

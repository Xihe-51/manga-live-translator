# -*- coding: utf-8 -*-
"""抓屏 + 翻页检测（imagehash / frame_diff 两种判定）。"""
import time

import cv2
import imagehash
import mss
import numpy as np
from PIL import Image


class ScreenCapturer:
    """持有 mss 抓屏实例，供 GUI 和命令行复用。"""

    def __init__(self):
        self._sct = mss.mss()

    def grab(self, region):
        """抓取屏幕区域 region=[左,上,宽,高]，返回 BGR 图。"""
        monitor = {"left": region[0], "top": region[1],
                   "width": region[2], "height": region[3]}
        shot = self._sct.grab(monitor)
        return np.array(shot)[:, :, :3]  # BGRA -> BGR

    @staticmethod
    def hash(bgr):
        """计算图像的感知哈希（用于翻页判定）。"""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return imagehash.average_hash(Image.fromarray(rgb))

    def close(self):
        self._sct.close()


def diff_ratio(a_bgr, b_bgr, pixel_threshold=25):
    """两帧的帧差比例：灰度差 > pixel_threshold 的像素占总像素的比例。"""
    ga = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    if ga.shape != gb.shape:
        return 1.0
    diff = cv2.absdiff(ga, gb)
    return float(np.count_nonzero(diff > pixel_threshold)) / diff.size


def run_capture_loop(cfg, out, process, on_page=None):
    """实时抓屏循环：画面稳定且与已处理页不同时，触发一次翻译。

    process(image_bgr) -> 处理当前页；on_page(page_no, image_bgr) 可选回调。
    翻页判定按 page_change.method 分支：
    - imagehash：感知哈希距离 > page_change.threshold 视为变化（默认）
    - frame_diff：帧差比例 > page_change.diff_threshold 视为变化
    """
    cap = cfg.get("capture", {})
    interval = cap.get("interval_ms", 500) / 1000.0
    region = cap.get("region", [0, 0, 0, 0])
    page_cfg = cfg.get("page_change", {})
    method = page_cfg.get("method", "imagehash")
    threshold = page_cfg.get("threshold", 8)
    diff_threshold = page_cfg.get("diff_threshold", 0.02)
    stable_needed = 3

    if len(region) != 4 or region[2] <= 0 or region[3] <= 0:
        print("请先设置抓屏区域：config.yaml 的 capture.region = [左, 上, 宽, 高]")
        return

    capturer = ScreenCapturer()

    def snapshot(bgr):
        """给当前帧打快照：imagehash 模式附带感知哈希。"""
        if method == "frame_diff":
            return (bgr, None)
        return (bgr, capturer.hash(bgr))

    def changed(a, b):
        """两帧是否算变化（超过对应阈值）。"""
        if method == "frame_diff":
            return diff_ratio(a[0], b[0]) > diff_threshold
        return (b[1] - a[1]) > threshold

    prev = None          # 上一帧快照
    processed = None     # 已处理页快照
    stable_frames = 0
    page_no = 0
    print(f"开始实时抓屏，区域={region}，翻页判定={method}，Ctrl+C 退出")
    try:
        while True:
            bgr = capturer.grab(region)
            cur = snapshot(bgr)

            if prev is not None and not changed(prev, cur):
                stable_frames += 1
            else:
                stable_frames = 1
            prev = cur

            if stable_frames >= stable_needed and (processed is None or changed(processed, cur)):
                processed = cur
                page_no += 1
                print(f"\n===== 检测到第 {page_no} 页，开始翻译 =====")
                process(bgr)
                if on_page:
                    on_page(page_no, bgr)

            time.sleep(interval)
    finally:
        capturer.close()
# -*- coding: utf-8 -*-
"""实时阅读的纯状态判定：变化即清旧结果，稳定后处理，慢速滚动不累计误判。"""
import hashlib
import cv2
import numpy as np


class FrameTracker:
    def __init__(self, stable_frames=2, pixel_threshold=20, change_ratio=0.003):
        self.stable_needed = max(2, int(stable_frames))
        self.pixel_threshold = max(1, int(pixel_threshold))
        self.change_ratio = max(0.0001, float(change_ratio))
        self.reset()

    def reset(self):
        self.anchor = None
        self.revision = 0
        self.stable_frames = 0

    def observe(self, bgr):
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
            raise ValueError("抓屏返回空画面")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 480 / max(gray.shape))
        sample = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                   max(1, round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        changed = self.anchor is None or self.anchor.shape != sample.shape
        if not changed:
            diff = cv2.absdiff(self.anchor, sample)
            changed = np.count_nonzero(diff >= self.pixel_threshold) / diff.size >= self.change_ratio
        if changed:
            self.anchor = sample.copy()
            self.revision += 1
            self.stable_frames = 1
        else:
            self.stable_frames += 1
        return changed, self.stable_frames >= self.stable_needed, self.revision


def frame_key(bgr):
    """完整画面精确键；近似哈希只判变化，不用于复用文字位置不同的译文。"""
    data = np.ascontiguousarray(bgr)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(data.shape).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()

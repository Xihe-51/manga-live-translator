# -*- coding: utf-8 -*-
"""对比 medium 与 mobile 模型在横排/竖排/密集图上的耗时与识别结果。

用法：python tests/compare_ocr_models.py
"""
import gc
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMMON = {
    "lang": "japan",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": True,
    "text_det_limit_side_len": 960,
    "enable_mkldnn": False,   # 本机 PaddlePaddle 3.3 的 oneDNN 路径会抛 NotImplementedError
}

SETUPS = (
    ("现状 medium", {}),
    ("mobile 模型", {"text_detection_model_name": "PP-OCRv5_mobile_det",
                     "text_recognition_model_name": "PP-OCRv5_mobile_rec"}),
)


def read_texts(ocr, image):
    result = ocr.predict(image)
    texts = []
    for res in (result or []):
        try:
            values = res["rec_texts"]
        except Exception:
            values = getattr(res, "rec_texts", []) or []
        texts.extend(str(v) for v in (values or []))
    return texts


def build_samples():
    samples = []
    for path in ("examples/test.png", "examples/test_vertical.png"):
        image = cv2.imread(path)
        if image is not None:
            samples.append((os.path.basename(path), image))
    base = cv2.imread("examples/test.png")
    if base is not None:
        # 密集图：2x2 平铺并缩小，模拟一页里有多个气泡
        tile = cv2.resize(base, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        dense = np.full((tile.shape[0] * 2, tile.shape[1] * 2, 3), 255, dtype=np.uint8)
        for row in range(2):
            for col in range(2):
                dense[row * tile.shape[0]:(row + 1) * tile.shape[0],
                      col * tile.shape[1]:(col + 1) * tile.shape[1]] = tile
        samples.append(("密集2x2", dense))
    return samples


def main():
    samples = build_samples()
    print("样本：", ", ".join(f"{n}({i.shape[1]}x{i.shape[0]})" for n, i in samples))
    outcomes = {}
    for name, extra in SETUPS:
        from paddleocr import PaddleOCR
        options = dict(COMMON)
        options.update(extra)
        print(f"\n===== {name} =====")
        start = time.perf_counter()
        ocr = PaddleOCR(**options)
        print(f"  构建 {time.perf_counter() - start:.1f}s")
        result = {}
        try:
            for label, image in samples:
                read_texts(ocr, image)  # 预热
                times = []
                for _ in range(2):
                    t0 = time.perf_counter()
                    texts = read_texts(ocr, image)
                    times.append(time.perf_counter() - t0)
                seconds = min(times)
                result[label] = texts
                print(f"  {label}: {seconds:.2f}s | {len(texts)} 块")
                print(f"    {texts}")
        finally:
            del ocr
            gc.collect()
        outcomes[name] = result

    print("\n===== 结果差异（mobile 相对 medium）=====")
    names = list(outcomes)
    if len(names) == 2:
        left, right = outcomes[names[0]], outcomes[names[1]]
        for label in left:
            same = sorted(left[label]) == sorted(right.get(label, []))
            print(f"  {label}: {'完全一致' if same else '不一致'}")
            if not same:
                print(f"    medium: {left[label]}")
                print(f"    mobile: {right.get(label)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

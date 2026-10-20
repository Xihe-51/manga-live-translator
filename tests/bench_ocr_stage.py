# -*- coding: utf-8 -*-
"""定位 OCR 的固定开销：mkldnn 在本机 PaddlePaddle 3.3 不可用，改从线程数、
方向分类器、检测边长、模型档位找速度，并用空图测每次调用的固定成本。

用法：python tests/bench_ocr_stage.py [样图]
"""
import gc
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = {
    "lang": "japan",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": True,
    "text_det_limit_side_len": 960,
    "enable_mkldnn": False,
}

VARIANTS = (
    ("A 现状", {}),
    ("A2 关竖排方向分类", {"use_textline_orientation": False}),
    ("A3 线程=8", {"cpu_threads": 8}),
    ("A4 检测边长 640", {"text_det_limit_side_len": 640}),
    ("A5 mobile 模型", {"text_detection_model_name": "PP-OCRv5_mobile_det",
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


def timeit(ocr, image, rounds=2):
    read_texts(ocr, image)  # 预热
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        texts = read_texts(ocr, image)
        times.append(time.perf_counter() - start)
    return min(times), texts


def run(name, extra, samples):
    from paddleocr import PaddleOCR
    options = dict(BASE)
    options.update(extra)
    print(f"\n===== {name} =====")
    if extra:
        print("  额外参数:", extra)
    try:
        start = time.perf_counter()
        ocr = PaddleOCR(**options)
        init = time.perf_counter() - start
    except Exception as error:
        print(f"  构建失败：{type(error).__name__}: {error}")
        return None
    print(f"  构建 {init:.1f}s")
    try:
        for label, image in samples:
            try:
                seconds, texts = timeit(ocr, image)
            except Exception as error:
                print(f"  {label}: 推理失败 {type(error).__name__}: {error}")
                return None
            print(f"  {label} ({image.shape[1]}x{image.shape[0]}): {seconds:.2f}s | {len(texts)} 块 | {texts}")
            if label == "空图 200x100":
                blank_cost = seconds
        return blank_cost
    finally:
        del ocr
        gc.collect()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/test.png"
    image = cv2.imread(path)
    if image is None:
        print("读图失败：", path)
        return 1
    blank = np.full((100, 200, 3), 255, dtype=np.uint8)
    big = cv2.resize(image, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
    samples = (("空图 200x100", blank), ("原图", image), ("放大1.7倍", big))

    print("CPU 核心数:", os.cpu_count())
    print("样图:", path, f"{image.shape[1]}x{image.shape[0]}")
    summary = []
    for name, extra in VARIANTS:
        blank_cost = run(name, extra, samples)
        summary.append((name, blank_cost))

    print("\n===== 汇总 =====")
    for name, blank_cost in summary:
        print(f"  {name}: 空图固定开销 {'失败' if blank_cost is None else f'{blank_cost:.2f}s'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""语联世界 —— 引擎包。

把 OCR、翻译、覆盖渲染、抓屏、单页编排拆成独立模块，
供命令行（main.py）和图形界面（ui/）共用。
"""
from .pipeline import Pipeline, process_image
from .ocr import OcrEngine
from .translate import Translator
from .capture import ScreenCapturer

__all__ = ["Pipeline", "process_image", "OcrEngine", "Translator", "ScreenCapturer"]
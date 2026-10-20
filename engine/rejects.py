# -*- coding: utf-8 -*-
"""失败样本留存：把被清洗掉、判为无效的模型输出按天落成 JSONL，方便事后回看。

翻译链路对无效输出是"宁缺毋滥"——不抹字、不渲染，但直接丢掉就看不清模型到底
吐了什么。这里在每个丢弃点补记一条（原因 + 原文 + 模型返回），按天分文件
（rejects-YYYYMMDD.jsonl）控制单文件体积。只存文本，不存图片。

两个约定与译文缓存一致：没有 rejects 配置节时整体停用（避免测试污染用户
数据）；任何写入失败只提示不抛错——留存只是排查手段，不能拖垮翻译本身。
"""
from __future__ import annotations

import datetime
import json
import os
import threading

# 单个字段最多存 2000 字符：模型整段思考过程吐出来时别把文件灌炸
_MAX_FIELD = 2000


class RejectLog:
    """按天分文件的 JSONL 追加器；翻译线程与设置线程并发调用安全。"""

    def __init__(self, cfg):
        section = (cfg or {}).get("rejects") or {}
        self.enabled = bool(section.get("enabled", False))
        self.dir = str(section.get("dir") or os.path.join("cache", "rejects"))
        self._lock = threading.Lock()

    def record(self, reason, source, model_output):
        """记一条无效样本；原文与模型返回全空时跳过，写入失败不影响翻译。"""
        if not self.enabled:
            return
        source = str(source or "").strip()
        model_output = str(model_output or "").strip()
        if not source and not model_output:
            return
        now = datetime.datetime.now()
        entry = {
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": str(reason or "未知"),
            "source": source[:_MAX_FIELD],
            "model": model_output[:_MAX_FIELD],
        }
        path = os.path.join(self.dir, "rejects-" + now.strftime("%Y%m%d") + ".jsonl")
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            with self._lock:
                os.makedirs(self.dir, exist_ok=True)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line)
        except OSError as error:
            print(f"  失败样本写入失败（不影响翻译）：{error}", flush=True)
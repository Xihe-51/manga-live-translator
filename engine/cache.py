# -*- coding: utf-8 -*-
"""译文持久缓存：把「原文 → 译文」对照落盘成 JSON，重读旧页时零调用出结果。

三个设计取舍：
  - 按 provider|base_url|model 分桶：换模型不会误用旧译文，切回来还能命中；
  - 没有 cache 配置节（如单元测试的极简配置）时整体停用，避免测试污染用户数据；
  - 任何读写失败只警告不抛错——缓存只是加速手段，不能拖垮翻译本身。
"""
from __future__ import annotations

import json
import os

_DEFAULT_PATH = os.path.join("cache", "translations.json")
_DEFAULT_MAX = 5000


class TranslationCache:
    """一次加载、增量更新；有变更才整体原子重写，避免半截 JSON。"""

    def __init__(self, cfg):
        section = (cfg or {}).get("cache") or {}
        self.enabled = bool(section.get("enabled", False))
        self.path = str(section.get("path") or _DEFAULT_PATH)
        self.max_entries = max(0, int(section.get("max_entries", _DEFAULT_MAX)))
        self._buckets = {}
        self._dirty = False
        if self.enabled:
            self._load()

    def lookup(self, signature, texts):
        """取出命中的译文，只返回确实有缓存的原文。"""
        if not self.enabled:
            return {}
        pairs = self._buckets.get(signature) or {}
        return {t: pairs[t] for t in texts if t in pairs}

    def remember(self, signature, mapping):
        """记录新译文并落盘；空译文和「原样返回」不入缓存，避免固化失败结果。"""
        if not self.enabled:
            return 0
        # 函数内导入：translate 模块反向依赖本模块，顶层导入会形成循环。
        from .translate import normalize_for_compare

        pairs = self._buckets.setdefault(signature, {})
        changed = 0
        for source, target in mapping.items():
            source = str(source)
            target = str(target or "").strip()
            if not source or not target:
                continue
            # 归一化后相等即"原样返回"的回声；只改标点/空格不算翻译。
            if normalize_for_compare(target) == normalize_for_compare(source):
                continue
            if pairs.get(source) != target:
                changed += 1
            pairs[source] = target
        if changed:
            if self.max_entries:
                # dict 保持插入顺序：超限时淘汰最早写入的条目。
                while len(pairs) > self.max_entries:
                    pairs.pop(next(iter(pairs)))
            self._dirty = True
            self.save()
        return changed

    def save(self):
        if not self.enabled or not self._dirty:
            return
        tmp = self.path + ".tmp"
        try:
            folder = os.path.dirname(self.path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "buckets": self._buckets}, handle, ensure_ascii=False)
            # 先写临时文件再替换：进程中途退出也不会留下损坏的缓存。
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as error:
            print(f"  译文缓存写入失败（不影响翻译）：{error}", flush=True)

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            print(f"  译文缓存读取失败，本次从空缓存开始：{error}", flush=True)
            return
        buckets = data.get("buckets") if isinstance(data, dict) else None
        if isinstance(buckets, dict):
            self._buckets = {str(k): dict(v) for k, v in buckets.items() if isinstance(v, dict)}
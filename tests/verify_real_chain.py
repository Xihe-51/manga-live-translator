# -*- coding: utf-8 -*-
"""真实日文 OCR / Ollama / 工作线程 / QWebChannel 集成验证（不截取桌面）。

运行：python -B tests/verify_real_chain.py --out-dir outputs/real_chain
--web 增加真实网页 DOM 验证；--raw-columns 关闭竖排列合并以对照旧行为。
默认验证真正的网页传输层，不以 Python 信号或本地模型日志代替回传证据。
仅调用配置里的本机 free 服务，输入为合成日文样图，不代表真实漫画验收。
"""
import argparse
import base64
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PIL import Image
from main import load_config
from engine import Pipeline
from ui.bridge import Bridge
from test_regressions import connect_channel


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


def wait_for(predicate, seconds=360):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(20)
    if not predicate():
        raise RuntimeError("真实链路验证等待超时")


def js(page, code):
    values = []
    page.runJavaScript(code, lambda value: values.append(value))
    wait_for(lambda: bool(values), 15)
    return values[0]


def snapshot(page):
    return json.loads(js(page, """JSON.stringify({
        pairs: Array.from(document.querySelectorAll('#pairs .pair')).map(el => ({
            source: el.querySelector('.src').textContent,
            translation: el.querySelector('.dst').textContent
        })),
        previewSrc: document.getElementById('preview').src,
        previewLoaded: document.getElementById('preview').complete && document.getElementById('preview').naturalWidth > 0,
        runDisabled: document.getElementById('run').disabled,
        logs: document.getElementById('log').textContent
    })"""))


def run(args, report):
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg["llm"]["provider"] = "free"
    if not cfg["llm"]["free"]["base_url"].startswith(("http://localhost:", "http://127.0.0.1:")):
        raise RuntimeError("本验证仅允许本机翻译服务")
    cfg.setdefault("ocr", {})["merge_vertical_columns"] = not args.raw_columns
    report.update({"python": sys.executable, "provider": "free", "model": cfg["llm"]["free"]["model"],
                   "merge_vertical_columns": not args.raw_columns, "web_dom_requested": args.web,
                   "scope": "真实模型和网页传输层；项目合成日文样图，不代表真实漫画验收", "cases": []})
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    start = time.perf_counter()
    pipeline = Pipeline(copy.deepcopy(cfg))
    report["model_load_seconds"] = round(time.perf_counter() - start, 3)
    bridge = Bridge(copy.deepcopy(cfg))
    bridge._pipeline = pipeline
    channel, transport, signals = connect_channel(bridge)
    view = None
    try:
        if args.web:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            view = QWebEngineView()
            view.resize(1000, 720)
            view.page().setWebChannel(channel)
            loaded = []
            view.loadFinished.connect(loaded.append)
            view.load(QUrl.fromLocalFile(str(ROOT / "ui/web/index.html")))
            view.show()
            wait_for(lambda: bool(loaded), 45)
            if not loaded[-1]:
                raise RuntimeError("验证用网页加载失败；不能宣称 DOM 端已通过，可用默认传输层模式验证")
            wait_for(lambda: js(view.page(), "typeof bridge !== 'undefined' && !!bridge"), 30)

        def dispatch(region):
            if view:
                js(view.page(), "bridge.translate_page(%s); true" % json.dumps(region))
            else:
                bridge.translate_page(region)

        def delivered():
            return [m for m in transport.messages if m.get("type") == 1]

        def results():
            return [json.loads(m["args"][0]) for m in delivered() if m.get("signal") == signals["result_ready"]]

        def logs():
            return [m["args"][0] for m in delivered() if m.get("signal") == signals["log_line"]]

        samples = [("horizontal", "test.png", ["おはよう", "今日はいい天気", "私は漫画が好き"]),
                   ("vertical", "test_vertical.png", ["おやすみなさい", "今日はとても楽しい一日"])]
        for name, filename, expected in samples:
            source = ROOT / "examples" / filename
            with Image.open(source) as image:
                width, height = image.size
            region = "0,0,%d,%d" % (width, height)
            bridge._selected_region = [0, 0, width, height]
            bridge._selected_png = source.read_bytes()
            offset, count, message_offset = len(logs()), len(results()), len(delivered())
            start = time.perf_counter()
            dispatch(region)
            wait_for(lambda: len(results()) > count and not bridge._busy)
            payload = results()[-1]
            if not payload.get("ok"):
                raise RuntimeError(payload.get("error", "未知处理错误"))
            target = args.out_dir / (name + ".png")
            target.write_bytes(base64.b64decode(payload["image"].split(",", 1)[1]))
            texts = [box["text"] for box in payload["boxes"]]
            case = {"name": name, "source": str(source), "output": str(target),
                    "elapsed_seconds": round(time.perf_counter() - start, 3), "expected_texts": expected,
                    "exact_matches": [text for text in expected if text in texts], "boxes": payload["boxes"],
                    "translations": payload["translations"],
                    "unchanged_translations": [src for src, dst in payload["translations"].items() if src == dst],
                    "progress": logs()[offset:], "result_delivered_to_webchannel": True,
                    "controls_restored": any(m.get("signal") == signals["busy_changed"] and m["args"] == [False] for m in delivered()[message_offset:]),
                    "text_set_exact": sorted(texts) == sorted(expected),
                    "translation_keys_exact": set(payload["translations"]) == set(texts),
                    "translation_values_nonempty": all(isinstance(dst, str) and dst.strip() for dst in payload["translations"].values()),
                    "completion_delivered": any("完成：共" in s for s in logs()[offset:])}
            if view:
                expected_pairs = [{"source": src, "translation": dst} for src, dst in payload["translations"].items()]
                wait_for(lambda: snapshot(view.page())["previewSrc"] == payload["image"] and snapshot(view.page())["previewLoaded"] and not snapshot(view.page())["runDisabled"], 15)
                wait_for(lambda: snapshot(view.page())["pairs"] == expected_pairs, 15)
                case["web_dom"] = snapshot(view.page())
                case["web_dom"].pop("previewSrc")
            report["cases"].append(case)
            print(json.dumps(case, ensure_ascii=False, indent=2), flush=True)
        bridge._selected_png = b"invalid PNG for error delivery test"
        count, offset = len(results()), len(logs())
        dispatch(region)
        wait_for(lambda: len(results()) > count and not bridge._busy)
        report["error_case"] = {"result": results()[-1], "error_delivered": any("出错：框选截图读取失败" in s for s in logs()[offset:])}
        if view:
            wait_for(lambda: "出错：框选截图读取失败" in snapshot(view.page())["logs"], 15)
            report["error_case"]["web_error_visible"] = True
        assert not results()[-1]["ok"] and report["error_case"]["error_delivered"]
        assert all(case["result_delivered_to_webchannel"] and case["completion_delivered"] and case["controls_restored"] for case in report["cases"])
        report["transport_passed"] = True
        report["web_dom_passed"] = True if view else None
        report["all_sample_texts_exact"] = all(case["text_set_exact"] for case in report["cases"])
        report["translation_coverage_passed"] = all(case["translation_keys_exact"] and case["translation_values_nonempty"] for case in report["cases"])
        report["translation_semantics"] = "需人工核对，不以有中文输出作为语义正确的自动判据"
    finally:
        bridge.shutdown()
        if view:
            view.close()
        app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/real_chain")
    parser.add_argument("--raw-columns", action="store_true")
    parser.add_argument("--web", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    original = sys.stdout
    with (args.out_dir / "run.log").open("w", encoding="utf-8") as log:
        sys.stdout = Tee(original, log)
        try:
            run(args, report)
            print("传输层成功/错误回传通过；所有预期原文精确匹配：", report["all_sample_texts_exact"])
            if not report["all_sample_texts_exact"] or not report["translation_coverage_passed"] or any(case["unchanged_translations"] for case in report["cases"]):
                raise SystemExit("识别或翻译质量未通过样图验收，详见 evidence.json")
        except BaseException as error:
            report["failure"] = str(error)
            raise
        finally:
            sys.stdout = original
            (args.out_dir / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

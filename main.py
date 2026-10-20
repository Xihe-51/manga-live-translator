# -*- coding: utf-8 -*-
"""漫画实时翻译 —— 程序入口。

单图模式：
    python main.py --image 原图.png --out 输出.png --provider mock

实时抓屏模式：
    python main.py --capture --region 0,0,900,500

图形界面：
    python main.py --gui
"""
import argparse
import os
import sys

import cv2

from engine import Pipeline, process_image
from engine.capture import run_capture_loop


def load_config(path="config.yaml"):
    import yaml
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"读不到配置文件：{path}")
        print("  确认路径是否正确；也可以不带 --config，直接用项目根目录的 config.yaml。")
        sys.exit(1)
    except yaml.YAMLError as error:
        print(f"配置文件 {path} 不是合法 YAML：{error}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="漫画实时翻译")
    ap.add_argument("--image", default=None, help="输入漫画原图路径（单图模式）")
    ap.add_argument("--capture", action="store_true", help="实时抓屏模式")
    ap.add_argument("--gui", action="store_true", help="启动原生设置页与透明实时阅读")
    ap.add_argument("--web-panel", action="store_true", help="使用旧网页调试面板（配合 --gui）")
    ap.add_argument("--region", default=None, help="抓屏区域 left,top,width,height（实时模式）")
    ap.add_argument("--out", default="output.png", help="输出中文漫画图路径")
    ap.add_argument("--config", default="config.yaml", help="配置文件路径")
    ap.add_argument("--provider", default=None, help="覆盖 llm.provider：free/paid/mock")
    ap.add_argument("--font-path", default=None, help="覆盖 render.font_path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.provider:
        cfg["llm"]["provider"] = args.provider
    if args.font_path:
        cfg.setdefault("render", {})["font_path"] = args.font_path

    # 图形界面
    if args.gui:
        # 供设置页把快捷键等改动写回同一个配置文件（保留注释，见 engine/configio.py）
        cfg["_config_path"] = os.path.abspath(args.config)
        from ui.app import run_gui
        run_gui(cfg, web_panel=args.web_panel)
        return

    # 实时抓屏模式
    if args.capture:
        if args.region:
            cfg.setdefault("capture", {})["region"] = [int(x) for x in args.region.split(",")]
        pipeline = Pipeline(cfg)

        def process(bgr):
            image_pil, boxes, _ = pipeline.process(bgr)
            image_pil.save(args.out)
            print(f"      翻译 {len(boxes)} 个文字块 -> {args.out}")

        run_capture_loop(cfg, args.out, process)
        return

    # 单图模式
    if not args.image:
        ap.error("单图模式需要 --image；实时模式用 --capture --region 左,上,宽,高；界面用 --gui")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"读图失败：{args.image}")
        sys.exit(1)

    image_pil, boxes, translations = process_image(image_bgr, cfg, args.out)
    vertical = sum(1 for b in boxes if b.get("vertical"))
    print(f"完成！检测 {len(boxes)} 个文字块（横排 {len(boxes) - vertical}，竖排 {vertical}），输出：{args.out}")


if __name__ == "__main__":
    main()
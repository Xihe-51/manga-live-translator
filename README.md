# 漫画实时翻译系统

PaddleOCR + OpenCV + LLM

边看日文漫画，边自动把气泡里的日文抹掉、回填中文，看起来像在直接看中文漫画。

支持三种运行形态：**单图处理**、**实时抓屏**、**图形界面**。

## 项目结构

```
fanyi/
├── main.py                    # 程序入口（单图 / 实时抓屏 / 图形界面 三种模式）
├── config.yaml                # 全局配置（抓屏、翻页、OCR、抹字、LLM、排版）
├── requirements.txt           # 依赖清单
├── engine/                    # 核心引擎包（命令行与图形界面共用）
│   ├── __init__.py            # 导出 Pipeline / process_image / OcrEngine / Translator / ScreenCapturer
│   ├── pipeline.py            # 单页编排：OCR → 抹字 → 翻译 → 回填
│   ├── ocr.py                 # PaddleOCR 常驻实例，检测文字块（横竖标记）
│   ├── compose.py             # 抹字：纯色覆盖 / 局部 inpaint / LaMa 深度修复
│   ├── translate.py           # 翻译：free / paid / mock 三通道
│   ├── render.py              # 排版回填：自适应字号 + 断行 + 居中（横排/竖排）
│   ├── hotkey.py              # 全局快捷键（Windows RegisterHotKey）
│   ├── export.py              # 译文对照导出（纯文本渲染）
│   └── capture.py             # mss 抓屏 + 翻页检测（imagehash / frame_diff）
├── ui/                        # 图形界面（PySide6 + WebEngine）
│   ├── __init__.py
│   ├── app.py                 # 界面入口 run_gui，后台预热 OCR
│   ├── bridge.py              # QWebChannel 桥 + 后台工作线程
│   ├── region_selector.py     # 全屏拖拽框选区域
│   └── web/                   # 前端面板（index.html / app.js / style.css）
├── examples/
│   ├── make_test_image.py     # 生成测试图脚本（横排 test.png / 竖排 test_vertical.png）
│   ├── test.png               # 测试输入图（横排 3 句）
│   └── test_vertical.png      # 测试输入图（竖排单列 / 双列）
└── output.png                 # 输出中文漫画图（运行后生成）
```

## 安装依赖

```bash
# 创建虚拟环境（可选）
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

> 注意：`paddleocr`/`paddlepaddle` 需 3.x 版本（适配 Python 3.13），且运行时关闭 oneDNN 加速（`enable_mkldnn=False`）以规避 PaddlePaddle 3.3 的 PIR 推理 bug。

> 图形界面（`--gui`）额外需要 **PySide6**，但当前 `requirements.txt` 尚未列出（仅在注释中提到）。使用图形界面前请手动安装：
>
> ```bash
> pip install PySide6
> ```

## 运行

### 单图模式

```bash
# mock 模式（原样回填，用于验证 OCR→抹字→排版链路，无需 LLM/key）
python main.py --image examples/test.png --out output.png --provider mock

# 免费本地 LLM（需先安装 Ollama 并 pull qwen2.5:7b）
python main.py --image examples/test.png --out output.png --provider free

# 付费云端 API（先在 config.yaml 填 api_key）
python main.py --image examples/test.png --out output.png --provider paid
```

### 实时抓屏模式

```bash
# 指定抓屏区域 [左,上,宽,高]，画面稳定并检测到翻页后自动翻译并保存
python main.py --capture --region 0,0,900,500
```

### 图形界面模式

```bash
python main.py --gui
```

界面上可切换翻译通道、点击「框选区域」全屏拖拽选定漫画区域，点击「翻译当前页」即可预览回填后的中文漫画。

### 全局快捷键（仅 Windows）

阅读时不用切窗口就能操作，避免每次去够那个悬浮小工具栏：

| 动作 | 默认快捷键 |
| --- | --- |
| 开始 / 暂停实时 | `Ctrl+Alt+S` |
| 重新框选区域 | `Ctrl+Alt+1` |
| 刷新当前页 | `Ctrl+Alt+2` |
| 回到设置页 | `Ctrl+Alt+3` |
| 退出透明阅读 | `Ctrl+Alt+4` |

在设置页点「快捷键」可以改绑。写法是「修饰键 + 一个主键」，例如 `ctrl+alt+s`；
留空表示不注册该动作。改完立即生效，但**重启后仍以 `config.yaml` 的 `hotkeys` 段为准**。

两个注意点：

- 某个组合被别的程序占用时，**只有那一个失效**，会在运行日志里明确说明，其余快捷键照常工作。
- 单键（如 `t`）和无主键的组合（如 `ctrl+alt`）会被拒绝——前者会抢走正常打字，后者没有意义。

### 导出译文对照

阅读时点工具栏的「导出」（或设置页的「导出译文」），会把译文写成纯文本对照表：

```
こんにちは
你好

さようなら
再见
```

原句一行、译句一行，段间空行——这样上下扫一眼就能看出哪句译错了，比左右对齐快得多。

几个行为：

- **实时阅读会累积翻过的每一页**，导出的是整段阅读的合集，不是当前这一屏；同一句重复识别只保留一次。
- 纯标点/符号碎片（OCR 常把边框线、省略号当文字）默认丢弃，可用 `export.drop_noise` 关掉。
- 文件名带时间戳，避免多次导出互相覆盖；同一秒内连点两次会自动加 `_2`、`_3` 后缀。
- 导出位置见运行日志（会打印完整路径）。

## 命令行参数

| 参数            | 说明                          |
| ------------- | --------------------------- |
| `--image`     | 输入漫画原图路径（单图模式）              |
| `--capture`   | 启用实时抓屏模式                    |
| `--gui`       | 启动图形界面                      |
| `--region`    | 抓屏区域 `左,上,宽,高`（实时抓屏模式用）     |
| `--out`       | 输出中文漫画图路径（默认 `output.png`）  |
| `--config`    | 配置文件路径（默认 `config.yaml`）    |
| `--provider`  | 覆盖翻译通道：`free`/`paid`/`mock` |
| `--font-path` | 覆盖中文字体路径                    |

## 测试

运行 mock 模式即可快速验证整条链路，无需任何模型：

```bash
python main.py --image examples/test.png --out output.png --provider mock
```

预期输出逐行打印检测到的文字块坐标、原文与置信度，以及"原文 → 译文"映射和各阶段用时，并在 `output.png` 生成抹字+回填后的中文漫画图。

## 当前状态

- **第一阶段（单图闭环）**：已完成。OCR → 抹字 → 翻译 → 排版回填全流程可用。

- **第二阶段（实时抓屏 + 图形界面）**：基础版已实现。抓屏循环、翻页检测、PySide6 图形界面与全屏框选均可用。

- **竖排文本**：已支持。OCR 保留 `h > w` 的竖排块并标记，渲染逐字纵排、多列自右向左。

- **LaMa 抹字**：已接入。`inpaint.method: lama` 基于 simple-lama-inpainting，首次运行自动下载 `big-lama.pt`（也可用 `inpaint.lama_model` 指定本地权重）。

- **全局快捷键**：已接入（仅 Windows）。开始/暂停、重选、刷新、设置、退出五个动作可在设置页改绑，冲突时单独降级不拖垮程序。

- **译文导出**：已接入。按「原句/译句」两行一组导出纯文本对照，实时阅读会累积翻过的每一页，便于整段校对。

- **frame\_diff 翻页检测**：已实现。`page_change.method: frame_diff` 按帧差像素占比 > `diff_threshold`（默认 2%）判定翻页。

- **识别质量兜底**：已接入。`ocr.min_score`（默认 0.6）丢弃低置信文字块并**保留原文不抹字**；模型若回一句「无法辨认」这类元回应（在评论输入而非翻译），判为无译文，不盖进漫画。

- **翻译质量兜底**：已接入。`llm.batch_size` 默认 2——批量一大，7B 模型就会串扰：把假名转写成罗马字（`こそり`→`kosoi`）、混进英文（`大丈夫大丈夫ぅ`→"没事没Thing"）、把邻句词意带过来（`幼馴染`→"小学同学"）。另有一道守卫：译文里冒出原文没有的拉丁字母时，该句自动整句重发一次。

- **术语表**：已接入。`config.yaml` 的 `glossary` 段可固定人名/专有名词译法（如 `幼馴染: 青梅竹马`），出现时强制采用；改词表会自动失效相关译文缓存。

- **竖列合并**：已改进。改用「字形尺寸接近 + 横向紧邻 + 纵向重叠 + 顶/底/居中至少一种对齐」判定，真实漫画页把碎片列合回整句（实测单页 43 块 → 24 块），避免把半句单独送去翻译。

## 文档

- [需求分析文档](漫画实时翻译系统_需求分析文档.md)

- [功能模块设计](漫画实时翻译系统_功能模块设计.md)

- [配置与接口说明](漫画实时翻译系统_配置与接口说明.md)


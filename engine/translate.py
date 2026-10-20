# -*- coding: utf-8 -*-
"""翻译：自动识别原文语言（日/英/韩/俄…），统一翻成简体中文。

免费（Ollama）/ 付费（DeepSeek 等 OpenAI 兼容）/ mock 三通道。

语言检测优先用 lingua 统计模型：纯汉字日文（人名、短句）靠字符区间计数无法
与中文区分，只能靠成词概率判断；没装 lingua 时降级为字符区间计数，行为与
旧版一致。
"""
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache

import requests

from .cache import TranslationCache
from .rejects import RejectLog

# 按 Unicode 区间统计字符所属文字系统，作为 lingua 缺席时的降级方案
_SCRIPTS = (
    ("kana", re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\uff66-\uff9f]")),
    ("hangul", re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")),
    ("han", re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")),
    ("latin", re.compile(r"[A-Za-z\u00c0-\u024f]")),
    ("cyrillic", re.compile(r"[\u0400-\u04ff]")),
)
LANGUAGE_NAMES = {"ja": "日文", "ko": "韩文", "en": "英文", "ru": "俄文", "zh": "中文", "unknown": "外文"}
# 批量翻译常见的 "1." / "10、" / "（3）" 编号前缀
_NUMBERING = re.compile(r"^\s*[(\[（【]?\s*(\d{1,3})\s*[)\]）】]?\s*([.、．)）:：])\s*")
# 统计模型只负责这几个我们真正会用到的语言
_LINGUA_LANGUAGES = ("CHINESE", "JAPANESE", "ENGLISH", "KOREAN", "RUSSIAN")
# 判定最低样本量：太短的文本（单个汉字、纯数字）统计模型没有意义，宁可交给规则
_MIN_DETECT_CHARS = 2
_MIN_CONFIDENCE = 0.55


def strip_numbering(line, index=None):
    """剥掉批量翻译返回的编号前缀；不动 "3.5 倍" 这类小数。

    传了 index（该行在批次里的 1 基序号）时按序号精确剥离：模型把译文本身
    以数字开头时（"8N"、"5.7.6"、"1个后台"）也剥得掉，而没传 index 时无法
    区分 "45. 2" 和 "3.5 倍"，只能保守保留。
    """
    match = _NUMBERING.match(line)
    if not match:
        return line
    rest = line[match.end():]
    if not rest:
        return line
    if index is not None:
        # 序号对得上才是编号行；对不上说明这行本来就以数字开头。
        return rest.strip() if int(match.group(1)) == index else line
    if match.group(2) == "." and rest[0].isdigit():
        return line
    return rest.strip()


# ===== 模型输出清洗 =====
# 本地小模型常把思考块、JSON 包装、markdown 装饰混进译文，甚至只回一段思考
# 过程。这里统一提取真正的译文；拿不到有效内容时返回空串，调用方按"该句无
# 译文"处理（不抹字、不渲染），宁缺毋滥。
_THINK_BLOCK = re.compile(r"(?is)\x3cthink(?:ing)?\x3e.*?\x3c/think(?:ing)?\x3e")
_THINK_OPEN = re.compile(r"(?is)<think.*$")          # 未闭合的思考块：其后全是思考
_ANALYSIS_BLOCK = re.compile(r"(?is)\x3canalysis\x3e.*?\x3c/analysis\x3e|</?analysis>")
# 思考泄漏关键词：出现即整段作废（模型把推理过程当译文吐了出来）
_THINK_LEAKS = ("thinking process", "analyze the request")
# 元回应/拒答关键词：模型没在翻译，而是在"评论"输入。实测 qwen2.5:7b 拿到
# OCR 垃圾（单个字母 "n"）时会回一句"无法辨认"——那是在复述系统提示词里
# "无法辨认的碎片…保留原文"这句，却被当成译文盖到漫画上。命中即判无译文，
# 保留原文总比盖一句"无法辨认"强。仅在结果很短时判命中，避免误伤正文
# （正常对白里出现"请提供"的场合）。
_META_RESPONSES = (
    "无法辨认", "无法识别", "无法翻译", "无法确定", "无法判断", "无法处理",
    "不能辨认", "不能识别", "不能翻译", "辨认不出", "识别不出",
    "请提供原文", "请提供内容", "请提供文本", "请提供需要", "未提供原文",
    "没有原文", "缺少原文", "原文缺失", "内容缺失", "文本缺失", "无有效内容",
    "i cannot translate", "i can't translate", "unable to translate",
    "cannot translate", "can't translate", "please provide the",
    "no text provided", "no input provided", "as an ai",
)
_META_MAX_CHARS = 24          # 超过这个长度的结果不再按"元回应"处理
# 拉丁字母（含带附加符号的西欧字母）。用来识别"译文里冒出原文没有的字母"：
# 小模型批量翻译时会把假名转写成罗马字（こそり→kosoi）、混进英文单词
# （大丈夫大丈夫ぅ→"没事没Thing"）。这类污染原文里没有，一抓一个准。
_LATIN_LETTERS = re.compile(r"[A-Za-z\u00c0-\u024f]")


def _number(value, default):
    """把配置里的值安全转成 float；缺失/非法返回 default。nan 也算非法。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number      # nan 不等于自身
# 残缺 JSON 的兜底提取：值引号可缺（被截断时抠到行尾）
_JSON_PAIR = re.compile(r'(?is)"(translation|translatedText)"\s*:\s*"((?:\\.|[^"\\])*)(?:"|$)')
# 开头的 JSON 外壳特征；壳在但抠不出内容时说明模型返回的是空/坏字段
_JSON_ENVELOPE = re.compile(r'(?is)^\s*\{\s*"(translation|translatedText|text|output|data)"\s*:')
_JSON_FIELDS = ("translation", "translatedText", "text", "output")
# 成对的包裹引号（ASCII 与全角弯引号）；日文「」『』是内容不是装饰，不动
_QUOTE_PAIRS = {'"': '"', "\u201c": "\u201d", "'": "'", "\u2018": "\u2019"}


def _strip_code_fence(text):
    """剥掉 ```json ... ``` 代码围栏；没有围栏时原样返回。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    body = "\n".join(lines).strip()
    return body or stripped


def _decode_json_fragment(fragment):
    """把 JSON 字符串片段里的转义还原成真实字符（\\n、\\uXXXX 等）。"""
    try:
        return json.loads(f'"{fragment}"')
    except ValueError:
        return fragment


def _pick_json_field(obj):
    """按常见字段名取译文，兼容 data.translations[0].translatedText。"""
    for key in _JSON_FIELDS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = obj.get("data")
    if isinstance(data, dict):
        items = data.get("translations")
        if isinstance(items, list) and items and isinstance(items[0], dict):
            value = items[0].get("translatedText")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_json_translation(text):
    """从完整或残缺的 JSON 里抠译文；抠不出返回 None。"""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            value = _pick_json_field(obj)
            if value:
                return value
    match = _JSON_PAIR.search(text)
    if match:
        value = _decode_json_fragment(match.group(2)).strip()
        if value:
            return value
    return None


def _strip_decorations(text):
    """剥掉包裹译文的 markdown 加粗与成对引号（可能嵌套，最多两轮）。"""
    result = text
    for _ in range(2):
        before = result
        if len(result) > 4 and result.startswith("**") and result.endswith("**"):
            result = result[2:-2].strip()
        if len(result) >= 2 and _QUOTE_PAIRS.get(result[0]) == result[-1]:
            result = result[1:-1].strip()
        if result == before:
            break
    return result


def is_meta_response(text):
    """判断模型是不是在"评论输入"而不是翻译（"无法辨认"、"请提供原文"）。

    这类回复会被当成正常译文渲染进漫画，比不翻更糟——原文被抹掉，盖上一句
    和剧情无关的话。只在结果很短时判定，正常对白偶尔含"请提供"不会误伤。
    """
    raw = str(text or "").strip()
    if not raw:
        return False
    if len(normalize_for_compare(raw)) > _META_MAX_CHARS:
        return False
    low = raw.lower()
    return any(marker in low or marker in raw for marker in _META_RESPONSES)


def has_latin_letters(text):
    """文本里是否含拉丁字母。"""
    return bool(_LATIN_LETTERS.search(str(text or "")))


def looks_contaminated(source, translation):
    """译文是否混入了原文没有的拉丁字母（批量串扰的典型症状）。

    原文本身有字母（"OK"、"GO"）时不判定，避免误伤；原文没有而译文冒出字母，
    基本就是模型把假名转写成了罗马字或塞进了英文单词，整句重发一次能救回来。
    """
    if not translation:
        return False
    return has_latin_letters(translation) and not has_latin_letters(source)


def sanitize_translation(text):
    """清洗模型返回值，返回有效译文；无效（空、省略号、思考泄漏）返回空串。

    批量翻译按行清洗时务必保留空占位，否则译文与原文的对应关系会错位。
    """
    if text is None:
        return ""
    raw = _THINK_OPEN.sub("", _THINK_BLOCK.sub("", str(text)))
    raw = _ANALYSIS_BLOCK.sub("", raw)
    raw = _strip_code_fence(raw)
    if not raw:
        return ""
    if any(marker in raw.lower() for marker in _THINK_LEAKS):
        return ""
    extracted = _extract_json_translation(raw)
    if extracted:
        raw = extracted
    elif _JSON_ENVELOPE.match(raw):
        # 是 JSON 壳但抠不出内容（空字段）：判无译文，别把壳当译文盖上去。
        return ""
    result = _strip_decorations(raw.strip())
    if not result or result in ("...", "…"):
        return ""
    if is_meta_response(result):
        # 模型在评论输入而非翻译：判无译文，让调用方保留原文。
        return ""
    return result


def normalize_for_compare(text):
    """比对用归一化：只留字母/数字/汉字/假名等实义字符，去掉标点与空白。

    用于识别"原样返回"的回声——模型只改标点或空格的"翻译"不算翻译。
    """
    return re.sub(r"[\W_]+", "", str(text or ""))


def is_garbled_translation(text):
    """判断译文是否明显是乱码/退化输出（重复舔字、U+FFFD、问号风暴等）。

    规则与 Kototoro 的 isLikelyGarbledText 对齐：命中即不抹字不渲染，保留
    原文总比盖上乱码强。
    """
    text = str(text or "").strip()
    if not text:
        return True
    if "\ufffd" in text:
        return True
    length = len(text)
    if length >= 4 and (text.count("?") + text.count("？")) * 2 >= length:
        return True
    normalized = normalize_for_compare(text)
    if not normalized:
        # 全是标点/符号（"..."、"！？"）：没有任何信息量。
        return True
    if len(normalized) >= 10:
        counts = sorted(Counter(normalized).values(), reverse=True)
        if counts[0] * 100 // len(normalized) >= 60:
            return True
        if (counts[0] + counts[1]) * 100 // len(normalized) >= 85:
            return True
    max_run = run = 1
    for i in range(1, length):
        if text[i] == text[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return length >= 6 and max_run * 3 >= length * 2


def _glossary_entries(cfg):
    """术语表：config.yaml 的 glossary 段（原文→译文）；空/非法配置返回空字典。"""
    section = (cfg or {}).get("glossary") or {}
    if not isinstance(section, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in section.items()
            if str(k).strip() and str(v).strip()}


def glossary_prompt(cfg):
    """术语表渲染成提示词片段；没有词表时返回空串。"""
    entries = _glossary_entries(cfg)
    if not entries:
        return ""
    return "术语表（原文→译文，出现时必须采用，不得改写）：" + "；".join(
        f"{source}→{target}" for source, target in entries.items())


# 微批次：批量提示越长，7B 级模型的串扰越重——把假名转写成罗马字
# （こそり→kosoi）、混进英文单词（大丈夫大丈夫ぅ→"没事没Thing"）、
# 把邻句的词意带过来。16 句真实漫画页对照（各跑两轮，含拉丁字母的句数）：
#   每批 1 句 0/0、2 句 0/0、4 句 1/1、8 句 1/2，耗时 6.2/4.7/4.2/3.8 秒。
# 取 2：污染归零，只比每批 8 句慢约 1 秒。可在 config.yaml 用 llm.batch_size 覆盖。
_BATCH_SIZE = 2              # 普通短句每批最多几个
_LONG_SENTENCE_CHARS = 28    # 达到这个长度的句子单独成批
_MIN_BATCH_SIZE = 1
_MAX_BATCH_SIZE = 32         # 上限：再大也只是把整页塞一批，反而更容易串扰


def _is_noise_text(text):
    """噪声句：去掉标点/空白后不足 2 个实义字符的碎片。

    这类碎片混在批量提示里容易被模型吞并或改写，单独发送代价最小。
    """
    return len(normalize_for_compare(text)) < 2


#: 假名 + 汉字。用来回答"这块里到底有没有日文内容"。
_KANA_OR_HAN = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\uff66-\uff9f"      # 假名
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")                   # 汉字


def has_japanese_content(text):
    """文本里是否含假名或汉字（判断它是不是日文对白）。纯符号、纯拉丁字母都不算。"""
    return bool(_KANA_OR_HAN.search(str(text or "")))


def is_untranslatable_noise(text):
    """日文漫画里既无假名也无汉字、且去掉符号后不足 2 字的块——OCR 噪声，不该送去翻译。

    实测真实页：拟声词「グッ」被读成 "$\\r$"（置信度 0.63，**刚好越过 ocr.min_score**），
    模型把它译成 "1." 盖了上去——真实内容被抹掉换成垃圾，比不翻更糟。
    置信度阈值挡不住它（0.63 > 0.6），但"这块里一个日文字都没有"是硬信号。
    注意只对日文页生效：英文漫画的纯拉丁块是正常内容，不能一起丢掉。
    """
    return not has_japanese_content(text) and _is_noise_text(text)


def split_batches(texts, size=_BATCH_SIZE):
    """把待翻句子切成小批：长句与噪声句单独成批，其余按 size 打包。"""
    batches = []
    current = []

    def flush():
        if current:
            batches.append(list(current))
            current.clear()

    for text in texts:
        if len(str(text)) >= _LONG_SENTENCE_CHARS or _is_noise_text(text):
            flush()
            batches.append([text])
        else:
            current.append(text)
            if len(current) >= size:
                flush()
    flush()
    return batches


@lru_cache(maxsize=1)
def _lingua_detector():
    """惰性构建并缓存 lingua 检测器；未安装时返回 None 由调用方降级。"""
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        return None
    languages = [getattr(Language, name) for name in _LINGUA_LANGUAGES]
    return LanguageDetectorBuilder.from_languages(*languages).build()


_LINGUA_CODES = {"CHINESE": "zh", "JAPANESE": "ja", "ENGLISH": "en", "KOREAN": "ko", "RUSSIAN": "ru"}


def _detect_by_lingua(text):
    """用 lingua 的成词概率判定语言；不可用/样本不足/置信度低时返回 None。"""
    detector = _lingua_detector()
    if detector is None:
        return None
    # 纯数字/符号没有语言信息，统计模型会强行给个答案，先挡掉。
    if not any(pattern.search(text) for _, pattern in _SCRIPTS):
        return None
    try:
        values = detector.compute_language_confidence_values(text)
    except Exception:
        return None
    for value in values:
        code = _LINGUA_CODES.get(value.language.name)
        if code is None:
            continue
        return code if value.value >= _MIN_CONFIDENCE else None
    return None


def _detect_by_script(texts):
    """字符区间计数（降级方案）：假名→日文、谚文→韩文，再看汉字/拉丁占比。"""
    counts = {name: 0 for name, _ in _SCRIPTS}
    for text in texts:
        text = str(text)
        for name, pattern in _SCRIPTS:
            counts[name] += len(pattern.findall(text))
    if counts["kana"]:
        return "ja"
    if counts["hangul"]:
        return "ko"
    if counts["han"] and counts["han"] >= counts["latin"]:
        return "zh"
    if counts["latin"]:
        return "en"
    if counts["cyrillic"]:
        return "ru"
    return "unknown"


def detect_language(texts):
    """自动判断原文语言，返回 "ja"/"ko"/"zh"/"en"/"ru"/"unknown"。

    先按字符区间拿一个"有没有假名/谚文"的硬信号：日文假名、韩文谚文是各自
    独有的，命中就直接采纳，不必劳烦统计模型。剩下的纯汉字场景（中文 vs
    日文人名）才交给 lingua 的成词概率判断——这正是区间计数做不到的地方。
    """
    texts = [str(t) for t in texts]
    script = _detect_by_script(texts)
    if script in ("ja", "ko", "ru", "unknown"):
        # 假名/谚文/西里尔各自独有，命中即定论；unknown 是纯符号，也别问了。
        return script
    # 到这里只剩「全是汉字/拉丁」两种情况。拉丁基本就是英文，不必问模型。
    if script == "en":
        return "en"
    # 纯汉字才是需要统计模型的地方：中文与日文共用汉字，字形上无法区分。
    # 注意结论有限——短人名（東堂容太）在两种语言里都成立，lingua 也无法判定，
    # 只有整句汉字文（"日本語の文章です"这种带假名的除外）里才有可观信号。
    # 所以阅读日漫的可靠做法仍然是手动固定 llm.source_lang=ja，不能指望这里。
    merged = "".join(texts).strip()
    if len(merged) < _MIN_DETECT_CHARS:
        return script
    return _detect_by_lingua(merged) or script


def language_name(code):
    return LANGUAGE_NAMES.get(code, LANGUAGE_NAMES["unknown"])


def build_models_url(base_url):
    """从 OpenAI 兼容 base_url 推导 /models 端点地址；推导不出来返回空串。

    兼容常见写法：带 /chat/completions 的完整端点、服务根地址、已带 /models
    的地址（原样保留）。
    """
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/models"):
        return base
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base + "/models"


def parse_model_ids(payload):
    """从 /models 响应里取模型 id，去重后按字母序排列；识别不了的结构返回空列表。"""
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    ids = set()
    for item in items:
        if isinstance(item, dict):
            value = item.get("id") or item.get("name") or item.get("model")
        else:
            value = item
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    return sorted(ids)


def fetch_models(endpoint, timeout=(5, 15)):
    """GET /models：返回 (实际地址, 模型 id 列表)；base_url 为空时抛 RuntimeError。"""
    url = build_models_url((endpoint or {}).get("base_url"))
    if not url:
        raise RuntimeError("该通道还没填 base_url，先在 config.yaml 里配好再测")
    key = (endpoint or {}).get("api_key")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return url, parse_model_ids(response.json())


# 默认提示词。可在 config.yaml 的 llm.prompts 段整体覆盖，用下面的占位符插值：
#   $from   原文语言（如"日文"）
#   $to     目标语言（固定为"简体中文"）
#   $detect 自动检测到的语言（同 $from，供模板作者区分"指定"与"识别"语义）
#   $text   原文（批量时是带序号的整段文本）
DEFAULT_SYSTEM_PROMPT = (
    "你是漫画翻译。把输入的$from对白翻译成$to，"
    "结合本批对白理解语气，使用自然口语；各条对应关系必须保持，不能合并或调换。"
    "输入来自OCR，可能有错字：按上下文合理意译，不要照抄可疑字符，"
    "也不要用罗马字转写假名（如 kosoi）、不要加括号注释或补充说明。"
    "确实无法辨认的碎片或单独拉丁字母不要编造成对白，保留原文。"
    "只输出译文，不要任何解释或额外内容。"
)
DEFAULT_BATCH_PROMPT = (
    "把下面每句$from翻译成$to。严格按行输出，每行只写译文，不要写序号、不要抄原文、不要解释：\n$text"
)
TARGET_LANGUAGE = "简体中文"


def _render_prompt(template, text="", source="", detected=""):
    """把 $变量 替换进模板。用 replace 而非 format，免得原文里的花括号炸掉。"""
    return (str(template)
            .replace("$text", str(text))
            .replace("$from", str(source))
            .replace("$detect", str(detected))
            .replace("$to", TARGET_LANGUAGE))


def _prompt_template(cfg, key, default):
    """取 config 里的提示词模板；空白串一律当没填，回退内置默认。"""
    override = ((cfg or {}).get("llm", {}).get("prompts", {}) or {}).get(key)
    if override is None or not str(override).strip():
        return default
    return override


def system_prompt(language, cfg=None, text=""):
    """系统提示词：优先用 config.yaml 的 llm.prompts.system，否则用内置默认。

    config.yaml 配了 glossary 时把术语表追加在末尾，固定译名不被模型自由发挥。
    """
    template = _prompt_template(cfg, "system", DEFAULT_SYSTEM_PROMPT)
    name = language_name(language)
    prompt = _render_prompt(template, text=text, source=name, detected=name)
    glossary = glossary_prompt(cfg)
    return prompt + "\n" + glossary if glossary else prompt


def batch_prompt(language, numbered, cfg=None):
    """批量翻译的用户提示词：同样可在 config 覆盖 llm.prompts.batch。"""
    template = _prompt_template(cfg, "batch", DEFAULT_BATCH_PROMPT)
    name = language_name(language)
    return _render_prompt(template, text=numbered, source=name, detected=name)


class Translator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cache = TranslationCache(cfg)
        self.rejects = RejectLog(cfg)

    @property
    def provider(self):
        return self.cfg["llm"].get("provider", "free")

    @property
    def batch_size(self):
        """每批最多几句；config.yaml 的 llm.batch_size，非法值回退内置默认。"""
        value = _number(self.cfg["llm"].get("batch_size"), _BATCH_SIZE)
        return int(min(max(value, _MIN_BATCH_SIZE), _MAX_BATCH_SIZE))

    def _cache_signature(self):
        """缓存分桶签名：换通道/服务地址/模型/术语表/提示词后都会失效旧译文。

        没配术语表、没改提示词时对应字段不参与签名，保证加词表、改模板之前的
        缓存仍然可用；改了就自动分到新桶，避免新旧配置的译文混用。
        """
        p = self.cfg["llm"][self.provider]
        signature = f"{self.provider}|{p.get('base_url')}|{p.get('model')}"
        glossary = _glossary_entries(self.cfg)
        if glossary:
            digest = hashlib.sha256(
                json.dumps(sorted(glossary.items()), ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:12]
            signature += f"|glossary:{digest}"
        # 用"生效的"提示词（内置默认或用户覆盖）算摘要。只算用户覆盖是不够的：
        # 内置默认提示词一改，老缓存会一直压着新提示词不生效。
        effective = {
            "system": _prompt_template(self.cfg, "system", DEFAULT_SYSTEM_PROMPT),
            "batch": _prompt_template(self.cfg, "batch", DEFAULT_BATCH_PROMPT),
        }
        digest = hashlib.sha256(
            json.dumps(effective, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        signature += f"|prompts:{digest}"
        return signature

    def source_language(self, texts):
        """原文语言：默认自动识别，可在 config.yaml 用 llm.source_lang 固定。"""
        override = str(self.cfg["llm"].get("source_lang", "auto") or "auto").strip().lower()
        if override and override != "auto":
            return override
        return detect_language(texts)

    def translate(self, texts, progress=None):
        """先小批翻译；批次失败或行数不匹配时逐句兜底并报告进度。"""
        notify = progress or (lambda message: None)
        texts = list(dict.fromkeys(t for t in texts if t))
        if not texts:
            return {}
        if self.provider == "mock":
            notify("测试通道：原样回填，不调用翻译服务")
            return {t: t for t in texts}

        language = self.source_language(texts)
        if language == "zh":
            # 已经是中文就没有可翻的，直接说清楚，别让模型把原文抄一遍。
            notify(f"自动识别：选区内 {len(texts)} 个文字块已经是中文，无需翻译")
            return {}
        notify(f"自动识别原文语言：{language_name(language)}，将翻译成简体中文")

        # 先查缓存：翻回已看过的页、重读旧内容时，命中的句子不再发给模型。
        signature = self._cache_signature()
        cached = self.cache.lookup(signature, texts)
        todo = [t for t in texts if t not in cached]
        if cached:
            notify(f"缓存命中 {len(cached)} 句" + (f"，只需翻译 {len(todo)} 句" if todo else "，全部直接使用缓存"))
        if not todo:
            return cached

        p = self.cfg["llm"][self.provider]
        url = p["base_url"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {p['api_key']}"}

        batches = split_batches(todo, size=self.batch_size)
        if len(batches) > 1:
            notify(f"正在翻译 {len(todo)} 句（{self.provider}，分 {len(batches)} 批），等待模型响应…")
        else:
            notify(f"正在批量翻译 {len(todo)} 句（{self.provider}），等待模型响应…")
        fresh = {}
        for index, batch in enumerate(batches, 1):
            if len(batches) > 1:
                notify(f"正在翻译第 {index}/{len(batches)} 批（{len(batch)} 句）…")
            lines = self._translate_batch(batch, url, headers, p, language, notify)
            fresh.update(dict(zip(batch, lines)))
        self._repair_contaminated(fresh, url, headers, p["model"], language, notify)
        notify(f"翻译完成：{len(todo)} 句")
        self.cache.remember(signature, fresh)
        result = {**cached, **fresh}
        # 归一化比较：模型只改标点/空格的"回声"同样算没翻出来。
        unchanged = [t for t, d in fresh.items()
                     if normalize_for_compare(d) == normalize_for_compare(t)]
        for text in unchanged:
            # 回声同样留样：模型没翻出来，事后回看能区分是提示词问题还是文本本身问题。
            self.rejects.record("原样返回", text, fresh.get(text, ""))
        if len(unchanged) == len(fresh):
            notify(f"模型把原文原样返回，没有产生译文；请确认选区是{language_name(language)}，"
                   "或换用付费通道重试")
        elif unchanged:
            notify(f"有 {len(unchanged)} 句译文与原文相同，已按原样显示")
        return result

    def _batch_translate(self, texts, url, headers, model, language="ja"):
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        user = batch_prompt(language, numbered, self.cfg)
        content = self._chat(url, headers, model, user, language)
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        # 按行号剥离，模型把译文以数字开头时（"8N"、"5.7.6"）才不会漏剥；
        # 再逐行清洗，无效行保留空占位以免译文与原文错位。
        cleaned = []
        for index, line in enumerate(lines, 1):
            value = sanitize_translation(strip_numbering(line, index))
            if not value:
                # 清洗后为空：这行是思考/JSON 壳/省略号，留样方便排查。
                self.rejects.record("清洗后为空", texts[index - 1] if index <= len(texts) else "", line)
            cleaned.append(value)
        return cleaned

    def _translate_one(self, text, url, headers, model, language="ja"):
        content = self._chat(url, headers, model, text, language)
        value = sanitize_translation(content)
        if not value:
            self.rejects.record("清洗后为空", text, content)
        return value

    def _translate_batch(self, batch, url, headers, p, language, notify):
        """翻一批：批量失败或行数不匹配时逐句兜底。

        超时、5xx、格式错误这些"整批级"故障逐句重发往往能救回来；401 与
        连不上服务不降级——逐句同样会失败，直接报错更快更清楚。
        """
        lines = None
        try:
            lines = self._batch_translate(batch, url, headers, p["model"], language)
        except requests.exceptions.ConnectionError:
            if self.provider == "free":
                raise RuntimeError(
                    "连不上本地 Ollama（http://localhost:11434）。"
                    "请确认已安装并启动 Ollama，且已执行 `ollama pull qwen2.5:7b`。"
                )
            raise RuntimeError(f"连不上翻译服务：{p['base_url']}")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                raise RuntimeError(
                    f"翻译接口鉴权失败（{self.provider}）。请检查 config.yaml 里 "
                    f"{self.provider}.api_key 是否填写正确。"
                )
            status = e.response.status_code if e.response is not None else "未知"
            notify(f"批量翻译失败（HTTP {status}），改为逐句翻译（共 {len(batch)} 句）")
        except RuntimeError as e:
            notify(f"批量翻译失败（{e}），改为逐句翻译（共 {len(batch)} 句）")
        if lines is None:
            return self._translate_each(batch, url, headers, p["model"], language, notify)
        if len(lines) != len(batch):
            notify(f"批量响应行数不匹配，改为逐句翻译（共 {len(batch)} 句）")
            return self._translate_each(batch, url, headers, p["model"], language, notify)
        return lines

    def _repair_contaminated(self, fresh, url, headers, model, language, notify):
        """把译文里冒出拉丁字母的句子整句重发一次（就地改 fresh）。

        批量上下文串扰的典型产物：假名被转写成罗马字（こそり→kosoi）、混进英文
        单词（大丈夫大丈夫ぅ→"没事没Thing"）。这些字母原文里没有，一抓一个准。
        整句单独重发基本能救回来，代价只有坏掉的那几句，不用为它牺牲整批速度。
        重发后仍有字母的（模型自己要坚持）就保留原结果，不做无谓的第三轮。
        """
        suspect = [text for text, value in fresh.items()
                   if looks_contaminated(text, value)]
        if not suspect:
            return
        notify(f"检测到 {len(suspect)} 句译文混入拉丁字母（批量串扰），改为逐句重翻")
        for text in suspect:
            fixed = self._translate_one(text, url, headers, model, language)
            if fixed and not has_latin_letters(fixed):
                fresh[text] = fixed

    def _translate_each(self, texts, url, headers, model, language, notify):
        """逐句兜底：批量失败或行数对不上时逐句重发，可读性优先于速度。"""
        lines = []
        for i, text in enumerate(texts, 1):
            notify(f"正在翻译第 {i}/{len(texts)} 句…")
            lines.append(self._translate_one(text, url, headers, model, language))
        return lines

    def _chat(self, url, headers, model, user_content, language="ja"):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt(language, self.cfg)},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
        }
        # config.yaml 的 llm.options 原样并入请求体（可覆盖 temperature 等默认值）。
        options = self.cfg["llm"].get("options") or {}
        if isinstance(options, dict):
            payload.update(options)
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=(5, 60))
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("翻译服务返回了空译文，请检查模型与服务配置")
            return content.strip()
        except requests.exceptions.Timeout as e:
            raise RuntimeError("翻译请求超时（连接限时 5 秒、读取限时 60 秒），请检查服务或缩小选区") from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise RuntimeError("翻译服务返回的格式不正确，应为 OpenAI 兼容的 chat/completions 响应") from e

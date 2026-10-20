# -*- coding: utf-8 -*-
"""配置写回：把设置页里的改动原样写回 config.yaml，同时保住注释与格式。

config.yaml 的注释本身就是使用说明，走 yaml.dump 会把它们全部抹掉，所以这里
采用"逐行就地替换"：只改目标段里目标键的值，行内注释、缩进、其他段落原样保留；
段内缺的键插到段末，整段缺失时在文件末尾补一段。写盘用临时文件 + os.replace，
与译文缓存同一套做法，中途出错不会留下半个文件。
"""
import os
import re

# 段头（顶格或缩进的 "key:"）；行首是 # 的注释行不会命中
_HEADER = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z_][\w-]*)\s*:.*$")
# 段内的 "key: 值  # 行内注释"；值是惰性匹配，注释要求前面有空白，
# 不会把值里（无空格紧跟）的 # 误当注释起点
_ENTRY = re.compile(r"^(?P<indent>[ \t]+)(?P<key>[A-Za-z_][\w-]*)(?P<sep>\s*:\s*)"
                    r"(?P<value>.*?)(?P<comment>[ \t]+#.*)?$")


def _quote(value):
    """写成 YAML 双引号标量；热键、术语这类简单值够用。"""
    text = str(value if value is not None else "")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _split_ending(line):
    """拆出行内容与行尾（\n / \r\n），替换时不改动原来的换行风格。"""
    body = line.rstrip("\r\n")
    return body, line[len(body):]


def update_section_values(path, section, values):
    """把 values 写入 config.yaml 的 section 段，返回实际写过的键列表。

    - 段内已有的键：原地替换值，保留行内注释与换行风格；
    - 段内没有的键：插到该段最后一行之后（与段内键同缩进）；
    - 整个段缺失：在文件末尾新建该段。
    内容没有实际变化时不写盘；写失败抛 OSError，由调用方决定降级策略。
    """
    values = {str(k): v for k, v in dict(values or {}).items()}
    if not values:
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    eol = "\r\n" if "\r\n" in text else "\n"

    lines = text.splitlines(keepends=True)
    header_index = None
    for index, line in enumerate(lines):
        body, _ = _split_ending(line)
        match = _HEADER.match(body)
        if match and match.group("key") == section:
            header_index = index
            break

    changed = []
    if header_index is None:
        # 整段缺失：文件末尾补一段，保持 ${section}: 顶格、键缩进两格。
        if text and not text.endswith(("\n", "\r")):
            text += eol
        block = [f"{section}:{eol}"]
        for key, value in values.items():
            block.append(f"  {key}: {_quote(value)}{eol}")
            changed.append(key)
        text += eol + "".join(block)
    else:
        header_indent = _HEADER.match(_split_ending(lines[header_index])[0]).group("indent")
        # 段体：段头之后、直到下一个缩进不大于段头的非空非注释行为止。
        body_end = len(lines)
        for index in range(header_index + 1, len(lines)):
            body, _ = _split_ending(lines[index])
            if not body.strip() or body.lstrip().startswith("#"):
                continue
            indent = body[:len(body) - len(body.lstrip(" \t"))]
            if len(indent) <= len(header_indent):
                body_end = index
                break
        # 插入点：段体里最后一个比段头缩进深的行之后（注释/空行前）。
        insert_at = header_index + 1
        entry_indent = None
        for index in range(header_index + 1, body_end):
            body, _ = _split_ending(lines[index])
            if not body.strip() or body.lstrip().startswith("#"):
                continue
            if len(body) - len(body.lstrip(" \t")) <= len(header_indent):
                continue
            insert_at = index + 1
            match = _ENTRY.match(body)
            if match and entry_indent is None:
                entry_indent = match.group("indent")
        pending = dict(values)
        for index in range(header_index + 1, body_end):
            body, ending = _split_ending(lines[index])
            match = _ENTRY.match(body)
            if not match or match.group("key") not in pending:
                continue
            key = match.group("key")
            rebuilt = (f"{match.group('indent')}{key}{match.group('sep')}"
                       f"{_quote(pending.pop(key))}{match.group('comment') or ''}")
            if rebuilt != body:
                lines[index] = rebuilt + ending
                changed.append(key)
        if pending:
            # 段内缺的键：用段内已有键的缩进插到段末；段里没有任何键时缩进两格。
            indent = entry_indent or header_indent + "  "
            added = []
            for key, value in pending.items():
                added.append(f"{indent}{key}: {_quote(value)}{eol}")
                changed.append(key)
            lines[insert_at:insert_at] = added
        text = "".join(lines)

    if not changed:
        return []
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return changed
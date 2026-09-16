#!/usr/bin/env python3
"""把飞书导出的 Markdown 转成适合 GitHub 渲染的 Markdown。

飞书 `drive +export --file-extension markdown` 会保留三类平台专有标记，
GitHub 不认识它们，直接提交会显示成裸标签：

  <title>…</title>                  文档标题 —— 改由文件名与 H1 表达
  <callout emoji="…">…</callout>    高亮块   —— 按语义映射成 GitHub Alerts
  https://my.feishu.cn/wiki/…       站内链接 —— 改成仓库内相对路径

单独使用：
    python3 convert.py <源 markdown> <输出 markdown> [--h1 标题] [--append 追加文件] [--demote N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))

# 高亮块语义 → GitHub Alerts 类型（NOTE 蓝 / TIP 绿 / IMPORTANT 紫 / WARNING 黄 / CAUTION 红）
# 正文本身已带「**这份地图的读法**」「**本章问题**」「**自检**」等前缀，表情符号无需另存
CALLOUT_TO_ALERT = {
    "📖": "NOTE",       # 卷首「这份地图的读法」
    "📌": "NOTE",       # 跨册对照说明
    "💡": "IMPORTANT",  # 每章开头的「本章问题」
    "💭": "TIP",        # 章末「自检」
    "⏳": "WARNING",    # 时效性提醒
}
DEFAULT_ALERT = "NOTE"

# 飞书 token（wiki node_token 与 obj_token 都收）→ 仓库内目标文件
TOKEN_TO_TARGET: dict[str, str] = {}
for _src in SOURCES:
    for _key in ("node", "obj"):
        if _src.get(_key):
            TOKEN_TO_TARGET[_src[_key]] = _src["out"]

TITLE_RE = re.compile(r"<title>.*?</title>\s*", re.S)
CALLOUT_RE = re.compile(r'<callout emoji="([^"]*)"\s*>\n(.*?)\n</callout>', re.S)
FEISHU_RE = re.compile(r"https?://[\w.-]*feishu\.cn/(?:wiki|docx|docs|sheets)/([A-Za-z0-9]+)")
HEADING_RE = re.compile(r"^(#{1,6})(\s+.*)$")

# 残留标签检查：先剔除行内代码与围栏代码块，避免把说明文字里的 `<callout>` 当残留
CODE_SPAN_RE = re.compile(r"`[^`\n]*`|```.*?```", re.S)
LEFTOVER_RE = re.compile(
    r"</?(?:callout|title|sheet|grid|bitable|whiteboard|mention-user|text_tag|sub-page-list)\b[^>]*>"
)


def apply_patches(text: str, patches: list[dict] | None) -> str:
    for patch in patches or []:
        if patch["from"] not in text:
            print(f"警告：补丁未命中，原文可能已改动 —— {patch['from'][:40]!r}", file=sys.stderr)
            continue
        text = text.replace(patch["from"], patch["to"])
    return text


def demote_headings(text: str, levels: int) -> str:
    """整篇标题下沉 N 级：README 自带的 H1 之下不应再出现 H1。"""
    if levels <= 0:
        return text
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = HEADING_RE.match(line) if not in_fence else None
        if m:
            hashes = "#" * min(len(m.group(1)) + levels, 6)
            out.append(hashes + m.group(2))
        else:
            out.append(line)
    return "\n".join(out)


def callout_to_alert(match: re.Match[str]) -> str:
    alert = CALLOUT_TO_ALERT.get(match.group(1), DEFAULT_ALERT)
    quoted = "\n".join(
        f"> {line}" if line.strip() else ">" for line in match.group(2).split("\n")
    )
    return f"> [!{alert}]\n{quoted}"


def rewrite_feishu_link(out_path: Path, match: re.Match[str]) -> str:
    target = TOKEN_TO_TARGET.get(match.group(1))
    if not target:
        return match.group(0)  # 不在本仓库范围内的飞书链接，原样保留
    return os.path.relpath(target, out_path.parent).replace(os.sep, "/")


def convert_text(raw: str, out_path: Path, spec: dict | None = None, append: Path | None = None) -> str:
    spec = spec or {}
    text = TITLE_RE.sub("", raw).lstrip("\n")
    text = apply_patches(text, spec.get("patches"))
    text = demote_headings(text, int(spec.get("demote") or 0))
    text = CALLOUT_RE.sub(callout_to_alert, text)
    text = FEISHU_RE.sub(lambda m: rewrite_feishu_link(out_path, m), text)

    if spec.get("h1"):
        text = f"# {spec['h1']}\n\n{text.lstrip()}"

    if append and append.exists():
        text = text.rstrip("\n") + "\n\n---\n\n" + append.read_text(encoding="utf-8").strip() + "\n"

    return re.sub(r"\n{3,}", "\n\n", text).rstrip("\n") + "\n"


def convert_file(raw: Path, out: Path, spec: dict | None = None, append: Path | None = None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    text = convert_text(raw.read_text(encoding="utf-8"), out, spec, append)
    out.write_text(text, encoding="utf-8")

    leftover = LEFTOVER_RE.findall(CODE_SPAN_RE.sub("", text))
    if leftover:
        print(f"警告：{out} 仍残留飞书标签 {sorted(set(leftover))}", file=sys.stderr)

    try:
        shown = out.relative_to(HERE.parent)
    except ValueError:
        shown = out
    print(f"  ✓ {shown}（{len(text)} 字符）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--h1", default=None)
    ap.add_argument("--append", type=Path, default=None)
    ap.add_argument("--demote", type=int, default=0)
    args = ap.parse_args()

    if not args.raw.exists():
        print(f"源文件不存在：{args.raw}", file=sys.stderr)
        return 1

    convert_file(args.raw, args.out, {"h1": args.h1, "demote": args.demote}, args.append)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

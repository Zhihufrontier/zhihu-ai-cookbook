#!/usr/bin/env python3
"""从飞书知识空间拉取「知乎 AI 蓝宝书」全部文稿并转换为 GitHub Markdown。

    python3 scripts/sync_from_feishu.py

流程：对 scripts/sources.json 里的每个节点调用 lark-cli 导出 markdown 到
临时目录，再交给 convert.py 做平台标记转换，最后清掉临时目录。
输出文件直接覆盖仓库内的对应路径，提交前用 git diff 检查即可。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import convert

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
STAGING = ROOT / ".staging"
LARK_CLI = os.environ.get("LARK_CLI", "/opt/homebrew/bin/lark-cli")
WIKI_URL = "https://my.feishu.cn/wiki/{node}"


def load_sources() -> list[dict]:
    return json.loads((SCRIPTS / "sources.json").read_text(encoding="utf-8"))


def export(node: str, stem: str) -> Path:
    """调用 lark-cli 把一篇飞书文档导出成 markdown，返回落盘路径。"""
    proc = subprocess.run(
        [
            LARK_CLI, "drive", "+export",
            "--url", WIKI_URL.format(node=node),
            "--file-extension", "markdown",
            "--output-dir", str(STAGING),
            "--file-name", stem,
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"lark-cli 导出失败（{node}）：{proc.stderr.strip() or proc.stdout.strip()}")

    raw = STAGING / stem
    if not raw.exists():
        # 个别情况下飞书会按文档标题落盘，兜底找最近生成的文件
        candidates = sorted(STAGING.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError(f"导出后未找到 markdown 文件（{node}）")
        raw = candidates[0]
    return raw


def main() -> int:
    if not (Path(LARK_CLI).exists() or shutil.which(LARK_CLI)):
        print(f"找不到 lark-cli：{LARK_CLI}（可用环境变量 LARK_CLI 覆盖）", file=sys.stderr)
        return 1

    sources = load_sources()

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    try:
        for i, src in enumerate(sources, 1):
            label = src.get("label", src["out"])
            print(f"→ [{i}/{len(sources)}] {label}")
            raw = export(src["node"], Path(src["out"]).name)
            convert.convert_file(
                raw,
                ROOT / src["out"],
                spec=src,
                append=(SCRIPTS / "readme-footer.md") if src["out"] == "README.md" else None,
            )
    finally:
        shutil.rmtree(STAGING, ignore_errors=True)

    print("✓ 同步完成，请用 git diff 检查后提交")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

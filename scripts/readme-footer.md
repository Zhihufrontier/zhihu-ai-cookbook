## 关于本仓库

这里是飞书知识空间「知乎 AI 蓝宝书」的只读镜像，用来做版本追踪与对外分发。

- 正文由 `scripts/sync_from_feishu.py` 从飞书直接导出，不做人工改写；飞书侧更新后重新跑一次脚本即可。
- 飞书的高亮块（`<callout>`）在导出后转为 GitHub 的提示块（blockquote alert），文字内容未变。
- 各册引用的篇目均链接回知乎站内原帖，内容版权归原作者所有。

## 同步与更新

重新拉取全部文稿并覆盖本地文件：

```bash
python3 scripts/sync_from_feishu.py
```

脚本依赖已授权的 `lark-cli`，会先导出到 `.staging/` 再逐个转换，结束后自动清理。转换规则写在 `scripts/convert.py`，文档与文件的对应关系（含必要的文本补丁）写在 `scripts/sources.json`。

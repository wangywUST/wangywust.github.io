# Resume metadata

本目录是中英文简历的单一数据源。每个 section 独立维护：

- `data/<section>.json`：双语结构化内容；
- `data/citations.bib`：论文信息的单一数据源；
- `scripts/<section>_section.py`：该 section 的生成入口；
- `generated/zh|en/<section>.tex|md`：自动生成的片段；
- `generated/resume_zh.md`、`generated/resume_en.md`：汇总后的 Markdown 简历。
- `generated/zh|en/publications.tex`：中英文简历论文列表；
- `generated/publications.md`：网站论文列表；
- `services.md`、`teaching.md`、`awards.md`：由对应 section JSON 完整生成；
- `index.md`：工作经历和教育经历区块由 JSON 生成，其余手写内容保留。

在仓库根目录运行：

```powershell
python MetaData/scripts/generate_all.py
```

也可只生成一个 section，例如：

```powershell
python MetaData/scripts/work_section.py
```

如需通过外部论文服务补充 BibTeX 的 URL/DOI，可单独运行：

```powershell
python MetaData/scripts/fill_bib_url.py
```

该辅助脚本会访问网络并改写 `data/citations.bib`，因此不会由 `generate_all.py` 自动调用。

请勿直接修改 `generated`、`services.md`、`teaching.md`、`awards.md` 中的生成内容，以及 `index.md` 的 `METADATA` 标记区块；简历信息修改对应 JSON，论文信息修改 `data/citations.bib`，然后重新运行生成器。

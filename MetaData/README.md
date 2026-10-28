# Resume metadata

本目录是中英文简历的单一数据源。每个 section 独立维护：

- `data/<section>.json`：双语结构化内容；
- `data/faculty_form.json`：教师信息统计表的文件名、栏目标题别名、更新策略和论文栏展示方式；
- `data/citations.bib`：论文信息的单一数据源；
- `scripts/<section>_section.py`：该 section 的生成入口；
- `generated/zh|en/<section>.tex|md`：自动生成的片段；
- `generated/resume_zh.md`、`generated/resume_en.md`：汇总后的 Markdown 简历。
- `generated/zh|en/publications.tex`：中英文简历论文列表；
- `generated/publications.md`：网站论文列表；
- `信息学院教师信息统计表_王艺炜_已更新.docx`：唯一的 Word 输出，直接在原表对应栏目中更新，不生成独立列表、预览或待替换 Word 文件。
- `generated/form_backups/*.zip`：更新前的压缩备份，解压后可恢复当时的统计表。
- `services.md`、`teaching.md`、`awards.md`：由对应 section JSON 完整生成；
- `index.md`：工作经历和教育经历区块由 JSON 生成，其余手写内容保留。

在仓库根目录运行：

```powershell
python MetaData/scripts/generate_all.py
```

首次使用先安装依赖：

```powershell
python -m pip install -r MetaData/requirements.txt
```

`generate_all.py` 默认直接填写 `信息学院教师信息统计表_王艺炜_已更新.docx`，无需再复制粘贴各个列表。
课程、科研经费、奖项及专业活动随数据源更新；空缺的栏目会按原表样式补入。学习和工作经历只在空缺时填写，已有的手写详细介绍、研究方向、招生文字、照片、办公信息等保留。
奖项沿用原表的阶段分组，专业活动保留已写明的 Action Editor、Senior Area Chair 等具体任职；补齐数据中的期刊编委、会议领域主席、Tutorial、会议及期刊审稿记录。
论文栏默认保留完整论文列表的网页链接；如需全部论文，修改 `data/faculty_form.json` 的 `publications_mode` 为 `full`。

填写前会以 ZIP 格式备份原文件到 `generated/form_backups/`。若 Word 正占用目标文件，会提示保存并关闭后重新运行，原文件保持不变，也不会生成 Word 副本。重复运行不会重复添加栏目、记录或编号。

也可只更新教师统计表（在仓库根目录运行）：

```powershell
python MetaData/scripts/faculty_form_word.py
```

如果当前目录已是 `MetaData`，上述路径去掉 `MetaData/` 前缀即可。
`scripts/publications_data.py` 只读取论文列表和其中的评级标注，不单独写 Word 文件。`scripts/<section>_section.py` 只更新相应模块的 TeX、Markdown；所有 Word 内容由 `faculty_form_word.py` 统一填入最终统计表。

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

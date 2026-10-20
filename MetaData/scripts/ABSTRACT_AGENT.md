# ABSTRACT_AGENT — 为 citations.bib 批量补 abstract 字段任务说明

本文件供 AI 代理执行：读取 `citations.bib` 里每一篇论文，只要对应的 PDF 存在，就读 PDF 首页文本，为该条目新增一个 `abstract` 字段，用 2–4 句英文讲清楚这篇论文的主要贡献。


***

## 1. 任务

1. 解析 `citations.bib`，列出所有条目（citekey + title + 已有字段）。
2. **增量过滤**：已经带 `abstract` 字段的条目直接跳过，不重写、不覆盖。
3. 对剩余条目，检查 `pdfs\{citekey}.pdf` 是否存在且合法（`%PDF` 头、≥20 KB）。
4. 对存在 PDF 的条目，提取 PDF 第一页（必要时第二页）文本，定位论文自带的 Abstract 段。
5. 基于提取到的文本（结合 title），用英文写一段 60–120 词的摘要，讲清 **问题 → 方法 → 主要贡献/结果**。
6. 把 `abstract = {...},` 字段外科式插入到 bib 条目里（不重排、不格式化其他字段）。
7. 校验改写后的 bib 仍能被 `bibtexparser` 正常解析。
8. 最后向用户汇报：新增补了几篇、跳过几篇（已有 abstract / 没 PDF / 文本提取失败），跳过清单。

## 2. 仓库路径（Windows，只有 PowerShell，没有 Bash）

| 用途 | 路径 |
|---|---|
| 仓库根目录 | `E:\wangywust.github.io` |
| bib 源文件（唯一权威，本次任务唯一允许修改的文件） | `E:\wangywust.github.io\MetaData\data\citations.bib` |
| PDF 存放目录 | `E:\wangywust.github.io\pdfs\` |
| 站点生成脚本（只读参考，禁止改动） | `E:\wangywust.github.io\MetaData\scripts\publications_markdown.py` |

* Python 3.13 可用，命令为 `python`。
* 依赖：`pypdf`（已安装，用于提取 PDF 文本）、`bibtexparser`（已安装，用于校验）。
* 临时脚本写到 `tmp\`，不要污染仓库结构。

## 3. 工作流程

### 3.1 解析 bib

用 `bibtexparser.load(..., parser=BibTexParser(customization=homogenize_latex_encoding))` 读入，得到所有条目。记录每个条目的 citekey、title，以及条目在原文中的起止位置（用于后面外科式插入）。

**不要用 `bibtexparser.dump()` 整体写回**——那会把整个文件的缩进、字段顺序、空行全部重排，造成巨大 diff。必须按 3.4 节的外科方式插入。

### 3.2 提取 PDF 首页文本

对每个条目 `k = citekey`，目标 PDF 是 `pdfs\{k}.pdf`。

```python
from pypdf import PdfReader
reader = PdfReader(pdf_path)
text = reader.pages[0].extract_text()
if len(text) < 500 and len(reader.pages) > 1:
    text += "\n" + reader.pages[1].extract_text()
```

注意：

* 有些 PDF 是扫描版图片，`extract_text()` 返回空串或乱码——这种条目标记为 "text_extract_failed"，跳过，不要瞎编摘要。
* 忽略 pypdf 对 annotation object 的非致命 warning。
* 本机 SSL/网络问题不影响这一步（本地文件）。

### 3.3 定位论文自带的 Abstract 段

在提取的文本里找 Abstract：

* 常见标记：`Abstract`、`ABSTRACT`、`Index Terms` 之前的那段、`1 Introduction` 之前的那段。
* 用正则 `r'(?i)abstract\s*(.+?)(?=\n\s*(?:1\s+introduction|I\.\s+introduction|introduction\b|index terms|keywords))'` 抓。
* 抓不到就把前 2000 字符喂给 LLM，让它自己判断 Abstract 在哪。

### 3.4 写摘要（LLM 完成，不是脚本）

对每篇论文，LLM 读：title + authors + venue + 提取到的 Abstract 段（或首页前 2000 字符），然后产出一段英文摘要。

**摘要写作规范**：

* 长度：60–120 词，2–4 句话。
* 结构：第一句讲问题/动机，第二句讲方法/核心 idea，第三句讲关键结果或贡献。
* 语气：客观、现在时、主动语态；不要写 "In this paper, we propose" 这种水话——直接说做了什么、发现了什么。
* 不要照抄论文自带 Abstract（通常 150–250 词太长），要压缩到核心贡献。
* 不要编造论文里没有的数字、SOTA 结论、数据集名；如果提取文本里没有具体数字，就不要写数字。
* 如果提取文本完全读不懂（扫描版乱码、首页全是参考文献），标记为 "text_extract_failed"，跳过。
* **增量规则**：如果该条目已经带 `abstract` 字段，直接跳过这一条，不重写、不覆盖。即使旧摘要看起来短或不准，也不要动——重跑任务时已经有 abstract 的条目视为"已完成"。

### 3.5 外科式插回 bib

**关键：不要整体重写 bib 文件。** 对每个要补 abstract 的条目，在原文里定位它的 `title = {...},` 那一行，在其后插入一行：

```
  abstract = {这里写摘要},
```

具体规则：

* 缩进与周围字段一致（2 个空格）。
* 字段风格对齐邻近条目：如果邻近是 `title = {` 就写 `abstract = {`，如果邻近是 `title={` 就写 `abstract={`。
* title 行一定带尾逗号（title 不可能是条目最后一个字段），所以插在它后面安全。
* 已经有 `abstract` 字段的条目在 3.4 节就被跳过了，不会走到这里。
* 不要改动任何其他字段、不要动字段顺序、不要动空行、不要重排条目。

### 3.6 BibTeX 转义

摘要文本里如果出现以下字符，必须转义或包在双花括号里：

* `&` → `\&`
* `%` → `\%`
* `#` → `\#`
* `_` → 单词内部保留原样（BibTeX 里 `_` 是下标，但普通英文摘要里一般没有）；如果真有，写成 `\_`。
* `{` `}` → 用 `\{` `\}` 或把整段包在 `{{...}}` 里。
* `\` 不要出现（摘要里不要写 LaTeX 命令，纯英文句子即可）。
* 双引号 `"` 在 BibTeX 字段值里通常 OK，但建议用 `''` 或直接去掉。

最稳妥的做法：写摘要时就避免这些字符，纯英文陈述句。

### 3.7 校验

全部插完后：

1. 用 `bibtexparser.load()` 重新读一遍 `citations.bib`，确认不报错。
2. 对比条目数：改前 N 条，改后仍为 N 条（不能多/不能少）。
3. 抽查 3–5 条，确认 `abstract` 字段内容正确、花括号配平、没有破坏相邻条目。

## 4. 已知障碍

* **扫描版 PDF**：`extract_text()` 返回空或乱码。这类条目跳过并报告，不要硬编摘要。
* **双栏排版**：pypdf 有时按列读导致句子顺序错乱。如果提取文本明显语序混乱（左右栏混在一起），标记为 `text_extract_failed`，跳过。
* **首页不是 Abstract**：有些会议（如 IEEE）把 Abstract 放在首页顶部，有些（如 ACL）放在标题后。抓不到正则就把前 2000 字符喂给 LLM 判断。
* **119 篇论文不要一次性塞进一个 prompt**：分批处理，每批 10–15 篇，每批结束后把这一批的 abstract 写入临时 JSON（`tmp\abstracts.json`），全部跑完再统一插回 bib。这样中途中断也不丢进度。

## 5. 收尾输出

1. 把所有 abstract 一次性外科式插入 `citations.bib`（见 3.5）。
2. 重新用 bibtexparser 校验。
3. **不生成报告文件**。最后向用户汇报：
   - 本次新增补了多少条 abstract；
   - 跳过清单（表格：`citekey`、标题、原因——已有 abstract 跳过 / 没 PDF / 扫描版乱码 / 文本提取失败）。
   - 支持重复跑：下次再执行时，已有 abstract 的条目会被自动跳过，只补新增的论文。

## 6. 硬性约束

* 只允许修改 `citations.bib` 这一个文件，且只能**新增 `abstract` 字段**；已有的 `abstract` 字段一律不动，不能动其他字段、不能删条目、不能重排。
* 禁止：改动 `pdfs\` 下任何 PDF、改动其他 `.md`/`.py`/`.yml` 文件、改动网站生成脚本。
* 摘要内容必须忠实于 PDF 提取到的文本；读不懂就标失败，**不许编造**。
* 交付前核对：bib 仍可被 bibtexparser 正常解析；条目总数与改动前一致；每个新加 abstract 的条目花括号配平；已有 abstract 的条目内容逐字节未变。

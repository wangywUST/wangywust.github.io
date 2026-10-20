# DOWNLOAD\_AGENT — 缺失论文 PDF 下载任务说明

本文件供 AI 代理（如 ChatGPT）执行：把仓库里缺失的论文 PDF 下载补齐，最后只汇报**哪些文件下载不下来**。



***

## 1. 任务



1. 以 `citations.bib` 为准校正 `paper_pdf_manifest.csv`（对账：删幽灵行、补漏行、去重、对齐标题）。

2. 找出所有缺失的论文 PDF。

3. 为每一篇缺失的论文寻找可用的开放下载源，下载到 `pdfs\` 目录。

4. 更新 `pdfs\paper_pdf_manifest.csv` 并再次对账。

5. 最后向用户输出：**哪些文件无法下载**（附原因和建议途径）。

## 2. 仓库路径（Windows，只有 PowerShell，没有 Bash）



| 用途             | 路径                                                             |
| -------------- | -------------------------------------------------------------- |
| 仓库根目录          | `E:\wangywust.github.io`（git 管理，学术主页）                          |
| **bib 源文件（citekey + 标题的唯一权威）** | `E:\wangywust.github.io\MetaData\data\citations.bib`          |
| PDF 存放目录       | `E:\wangywust.github.io\pdfs\`                                 |
| 下载状态清单         | `E:\wangywust.github.io\pdfs\paper_pdf_manifest.csv`           |
| 现有下载脚本（可参考其逻辑） | `E:\wangywust.github.io\MetaData\scripts\download_bib_pdfs.py` |



* Python 3.13 可用，命令为 `python`。

* 临时脚本写到 `tmp\` 或临时目录，不要污染仓库结构。

## 3. 先与 citations.bib 对账（强制步骤，不可跳过）

**`citations.bib` 是论文清单和标题的唯一权威源**。`paper_pdf_manifest.csv` 只是下载状态的缓存，不能作为论文清单的依据——它可能过时、漏行、错挂 citekey。动手下载之前，必须先用 bib 把 manifest 校正一遍。

manifest 列结构固定：



```
citekey,title,source_url,pdf_url,filename,status,message
```



### 3.1 解析 citations.bib

用正则 `@\w+\{([^,]+),` 抓出 bib 里所有条目，得到权威的 `citekey → title` 映射集合 B，同时保留每个条目的 `url`/`doi` 字段备用。bib 里每个条目就是一篇独立论文，**一个 citekey 只对应一篇论文、一个 PDF 文件**。

### 3.2 校正 manifest（按以下规则逐条核对并修复）

设 manifest 当前行集合为 M，citekey 集合为 K_M。

1. **幽灵行删除**：若某行 `citekey ∉ B`（bib 里根本没有这个 citekey），直接从 manifest 删掉该行。
2. **漏行补齐**：若某 `citekey ∈ B` 但 `∉ K_M`，在 manifest 新增一行；`title` 必须照抄 bib 里的 `title` 字段，`source_url` 优先用 bib 的 `url`/`doi`，`pdf_url` 留空，`status=not_found`。
3. **重复 citekey 修复**：若 manifest 里同一个 `citekey` 出现多行，这是错误（不是"共用 PDF"）。逐行核对标题：
   - 哪一行的标题与 bib 里该 citekey 的 `title` 一致，保留它；
   - 其余行是**错挂 citekey 的别的论文**——按它们的 `title` 在 bib 的 B 里反查真正的 citekey（标题精确匹配），把这些行的 citekey 改成正确值；
   - 若某行标题在 bib 里找不到对应条目，说明它是幽灵行，删掉并向用户报告。
4. **标题对齐**：若某行 `citekey ∈ B` 但标题与 bib 的 `title` 不一致，以 bib 为准覆盖 manifest 的 `title`（manifest 里的标题可能是旧的/错的）。
5. **PDF 文件名**：始终是 `pdfs\{citekey}.pdf`，不依赖 manifest 的 `filename` 列。

### 3.3 判定缺失

对账完成后，逐行检查 `pdfs\{citekey}.pdf` 是否存在且合法（`%PDF` 头、≥20 KB）；不存在就是缺失项，进入第 4 节下载。

注意：



* 执行前先重新扫描 `pdfs\` 目录（用户可能正手动下载），每篇下载前再确认一次文件不存在。

* **citekey 在 bib 里不可能重复**；若 manifest 里出现重复，一定是 3.2 节要修的错误，不要"只下一份然后在报告里注明"。

* bib 里的 `doi`/`url` 字段本身可能有错（历史上出现过把 A 论文的 DOI 抄到 B 条目下的情况）。代理**不得修改 citations.bib**（见第 7 节硬性约束），但发现 DOI 与真实论文对不上时，应在最终汇报里提醒用户，并用第 4 节搜索到的真实 DOI 作为下载源。

## 4. 逐篇下载



1. 从 manifest 取该行的 `title` 和 `source_url`。

2. 按优先级找可用的开放 PDF 源：

* **arXiv**：查 `https://export.arxiv.org/api/query?search_query=ti:"{标题}"&max_results=3`，取返回的 arXiv 编号，下载 `https://arxiv.org/pdf/{编号}.pdf`。间隔 ≥2-3 秒。

* **Semantic Scholar**：查 `https://api.semanticscholar.org/graph/v1/paper/search?query={标题}&fields=title,openAccessPdf,externalIds,venue,year&limit=3`，用返回的 `openAccessPdf.url`。间隔 ≥3 秒（无 key 会 429 限流，退避重试一次）。

* **ACL Anthology**：原 `source_url` 的 DOI 是 `10.18653/v1/...` 时，直接下载 `https://aclanthology.org/{论文id}.pdf`。

* **全网搜索**：搜「标题 + arXiv」「标题 + pdf」，找作者主页、会议官方 proceedings（NeurIPS、PMLR、ICML、ICLR、ACL、CVPR 等）。

1. 下载到 `pdfs\{citekey}.pdf`（先写 `.tmp`，校验通过后再改名落地）。

2. 校验：文件头以 `%PDF` 开头、大小 ≥20 KB、尽量与标题匹配（能提取首页文本比对关键词更好）。

3. 成功 → manifest 该行 `status=exists`；失败 → 保留失败状态，`message` 写明原因。

## 5. 已知障碍（避免重复踩坑）



* **OpenReview（**[openreview.net](https://openreview.net)**）直连必失败**（Cloudflare JS 挑战，脚本请求一律 403），不要浪费时间直连，直接换源：优先 arXiv，其次作者主页、会议官方 proceedings。

* **DOI 大多付费墙**（IEEE/ACM/Springer 返回 HTML 或 403）；只有 `10.18653/v1/...`（ACL Anthology）可直接下载。

* **本机 SSL 证书**：`urllib` 默认校验可能报 `CERTIFICATE_VERIFY_FAILED`，先 `import certifi`，不行就用 `ssl._create_unverified_context()` 兜底。

* 同一 IP 不要并发猛打 arXiv / Semantic Scholar，容易被限流。

## 6. 收尾输出



1. 全量重写 `pdfs\paper_pdf_manifest.csv`（schema 不变，`status` 用 `exists` / `download_failed` / `not_found`）。重写后必须再次跑一遍 3.2 节的对账校验，确认：
   - manifest 的 citekey 集合与 `citations.bib` 的 citekey 集合**完全相等**（无幽灵、无遗漏）；
   - 没有任何 citekey 在 manifest 里出现超过一次；
   - 每一行的 `title` 与 bib 里该 citekey 的 `title` 一致。

2. **不生成下载报告文件**。

3. 最后只向用户输出：**无法下载的清单**（表格：`citekey`、标题、原因、建议途径 / URL）；下载成功只需一句话带过数量，不用逐一列出。同时若发现 bib 里 DOI/url 与真实论文不符（如抄错），在表格下方单独提醒用户。

## 7. 硬性约束



* 只允许：向 `pdfs\` 新增 PDF 文件、重写 `paper_pdf_manifest.csv`。

* 禁止：修改 `citations.bib`、删除或覆盖仓库内任何既有文件、改动网站生成脚本。

* 来源可疑（内容与标题不匹配、来源不明）的 PDF 宁可不下载，如实列为 "无法下载"。

* 交付前核对：
  - manifest 的 citekey 集合 == `citations.bib` 的 citekey 集合，且无重复；
  - 每一行 `title` 与 bib 一致；
  - `status=exists` 的行在 `pdfs\` 下都有对应文件且 `%PDF` 头合法；
  - 无法下载的清单完整。
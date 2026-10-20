# DOWNLOAD\_AGENT — 缺失论文 PDF 下载任务说明

本文件供 AI 代理（如 ChatGPT）执行：把仓库里缺失的论文 PDF 下载补齐，最后只汇报**哪些文件下载不下来**。



***

## 1. 任务



1. 找出所有缺失的论文 PDF。

2. 为每一篇缺失的论文寻找可用的开放下载源，下载到 `pdfs\` 目录。

3. 更新 `pdfs\paper_pdf_manifest.csv`。

4. 最后向用户输出：**哪些文件无法下载**（附原因和建议途径）。

## 2. 仓库路径（Windows，只有 PowerShell，没有 Bash）



| 用途             | 路径                                                             |
| -------------- | -------------------------------------------------------------- |
| 仓库根目录          | `E:\wangywust.github.io`（git 管理，学术主页）                          |
| PDF 存放目录       | `E:\wangywust.github.io\pdfs\`                                 |
| 下载状态清单         | `E:\wangywust.github.io\pdfs\paper_pdf_manifest.csv`           |
| 现有下载脚本（可参考其逻辑） | `E:\wangywust.github.io\MetaData\scripts\download_bib_pdfs.py` |



* Python 3.13 可用，命令为 `python`。

* 临时脚本写到 `tmp\` 或临时目录，不要污染仓库结构。

## 3. 找出缺失的 PDF（与 paper\_pdf\_manifest.csv 比对）

`paper_pdf_manifest.csv` 是唯一的比对基准，每一行是一篇论文，列结构固定：



```
citekey,title,source_url,pdf_url,filename,status,message
```



* 第 1 列 `citekey`：论文的唯一标识，PDF 文件名就由它决定 → `pdfs\{citekey}.pdf`（例如 `ge2026prism` → `pdfs\ge2026prism.pdf`）。

* 第 2 列 `title`：论文标题，找源时用来搜索。

**判定缺失**：逐行检查 `pdfs\{citekey}.pdf` 是否存在，不存在就是缺失项，需要下载。

注意：



* 执行前先重新扫描 `pdfs\` 目录（用户可能正手动下载），每篇下载前再确认一次文件不存在。

* `citekey` 可能有重复（同一 key 对应两行 / 两篇论文，共用一个 PDF），只下载一份，输出时注明哪一行没有独立 PDF。

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



1. 全量重写 `pdfs\paper_pdf_manifest.csv`（schema 不变，`status` 用 `exists` / `download_failed` / `not_found`）。

2. **不生成下载报告文件**。

3. 最后只向用户输出：**无法下载的清单**（表格：`citekey`、标题、原因、建议途径 / URL）；下载成功只需一句话带过数量，不用逐一列出。

## 7. 硬性约束



* 只允许：向 `pdfs\` 新增 PDF 文件、重写 `paper_pdf_manifest.csv`。

* 禁止：修改 `citations.bib`、删除或覆盖仓库内任何既有文件、改动网站生成脚本。

* 来源可疑（内容与标题不匹配、来源不明）的 PDF 宁可不下载，如实列为 "无法下载"。

* 交付前核对：manifest 与 `pdfs\` 实际文件一一对应；下载的文件 `%PDF` 头合法；无法下载的清单完整。
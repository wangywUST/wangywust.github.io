import bibtexparser
from habanero import Crossref
import requests
import difflib
import re

# 从 DOI 获取 URL
def get_url_from_doi(doi):
    return f"https://doi.org/{doi}"

# 从 Arxiv ID 获取 URL
def get_url_from_arxiv(arxiv_id):
    return f"https://arxiv.org/abs/{arxiv_id}"

# 使用 Crossref API 获取 DOI 或 URL
def get_doi_from_title(title):
    cr = Crossref()
    try:
        result = cr.works(query=title, limit=3)
        for item in result['message']['items']:
            fetched_title = item.get('title', [''])[0]
            if compare_titles(title, fetched_title):
                return item.get('DOI', None)
    except Exception as e:
        print(f"Error fetching DOI for title '{title}': {e}")
    return None

# 标题相似度比较，利用 difflib 的相似度
def compare_titles(original_title, fetched_title, threshold=0.85):
    original_title = clean_title(original_title)
    fetched_title = clean_title(fetched_title)
    similarity = difflib.SequenceMatcher(None, original_title, fetched_title).ratio()
    return similarity >= threshold

# 清理标题中的特殊符号
def clean_title(title):
    return title.lower().replace('{', '').replace('}', '').replace('\n', '').strip()

# 改进后的 Arxiv ID 提取逻辑，从 journal 字段中提取
def extract_arxiv_id(entry):
    # 检查 'journal' 字段是否包含 Arxiv ID
    if 'journal' in entry:
        journal = entry['journal']
        arxiv_match = re.search(r'arxiv[:\s]*([\d.]+)', journal, re.IGNORECASE)
        if arxiv_match:
            return arxiv_match.group(1)  # 返回 Arxiv ID
    return None

# 解析 .bib 文件
def fill_bib_urls(bib_file):
    with open(bib_file, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    # 遍历每个条目
    for entry in bib_database.entries:
        if 'url' not in entry:
            # 先检查是否为 Arxiv 论文
            arxiv_id = extract_arxiv_id(entry)
            if arxiv_id:
                # 如果找到 Arxiv ID，使用 Arxiv URL
                entry['url'] = get_url_from_arxiv(arxiv_id)
                print(f"Arxiv URL added for {entry.get('title', 'No Title')}")
            else:
                # 如果没有 Arxiv ID，尝试使用 DOI
                doi = entry.get('doi', None)
                if doi:
                    entry['url'] = get_url_from_doi(doi)
                else:
                    # 如果没有 DOI，使用标题查询 DOI
                    title = entry.get('title', None)
                    if title:
                        doi = get_doi_from_title(title)
                        if doi:
                            entry['doi'] = doi
                            entry['url'] = get_url_from_doi(doi)
                        else:
                            print(f"DOI not found for title: {title}")

    # 保存修改后的 .bib 文件
    with open(bib_file, 'w', encoding='utf-8') as bibtex_file:
        bibtexparser.dump(bib_database, bibtex_file)

# 示例：自动填充 url 字段
fill_bib_urls('citations.bib')

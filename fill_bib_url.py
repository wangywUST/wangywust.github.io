import bibtexparser
from habanero import Crossref
import requests
import difflib

# 从 DOI 获取 URL
def get_url_from_doi(doi):
    return f"https://doi.org/{doi}"

# 使用 Crossref API 获取 DOI 或 URL
def get_doi_from_title(title):
    cr = Crossref()
    try:
        result = cr.works(query=title, limit=3)  # 增加查询条目，获取多个结果以进行匹配
        for item in result['message']['items']:
            fetched_title = item.get('title', [''])[0]
            if compare_titles(title, fetched_title):
                return item.get('DOI', None)  # 返回匹配的 DOI
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

# 解析 .bib 文件
def fill_bib_urls(bib_file):
    with open(bib_file, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    # 遍历每个条目
    for entry in bib_database.entries:
        if 'url' not in entry:
            doi = entry.get('doi', None)
            if doi:
                # 使用已有 DOI 填充 URL
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

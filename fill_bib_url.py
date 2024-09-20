import bibtexparser
from habanero import Crossref
import requests

# 从 DOI 获取 URL
def get_url_from_doi(doi):
    return f"https://doi.org/{doi}"

# 使用 Crossref API 获取 DOI 或 URL
def get_doi_from_title(title):
    cr = Crossref()
    try:
        result = cr.works(query=title, limit=1)
        if result['message']['items']:
            doi = result['message']['items'][0].get('DOI', None)
            return doi
    except Exception as e:
        print(f"Error fetching DOI for title '{title}': {e}")
    return None

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

    # 保存修改后的 .bib 文件
    with open('updated_' + bib_file, 'w', encoding='utf-8') as bibtex_file:
        bibtexparser.dump(bib_database, bibtex_file)

# 示例：自动填充 url 字段
fill_bib_urls('example.bib')

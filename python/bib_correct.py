#!/usr/bin/env python3
"""
高级BibTeX条目修正工具
自动查找缺失信息、更正标题、清理不必要字段
"""

import re
import sys
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.customization import *
import argparse
from urllib.parse import quote
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class AdvancedBibCorrector:
    """高级BibTeX条目修正器"""
    
    def __init__(self, search_online=True, remove_unnecessary=True):
        """
        初始化修正器
        
        Args:
            search_online: 是否启用在线搜索
            remove_unnecessary: 是否移除不必要的字段
        """
        self.search_online = search_online
        self.remove_unnecessary = remove_unnecessary
        
        # 定义必需字段
        self.required_fields = {
            'article': ['author', 'title', 'journal', 'year'],
            'book': ['author', 'title', 'publisher', 'year'],
            'inproceedings': ['author', 'title', 'booktitle', 'year'],
            'conference': ['author', 'title', 'booktitle', 'year'],
            'incollection': ['author', 'title', 'booktitle', 'publisher', 'year'],
            'phdthesis': ['author', 'title', 'school', 'year'],
            'mastersthesis': ['author', 'title', 'school', 'year'],
            'techreport': ['author', 'title', 'institution', 'year'],
            'misc': ['author', 'title', 'year'],
            'unpublished': ['author', 'title']
        }
        
        # 定义可选但有用的字段
        self.optional_fields = {
            'article': ['volume', 'number', 'pages', 'doi', 'url'],
            'book': ['isbn', 'edition', 'volume', 'series'],
            'inproceedings': ['pages', 'doi', 'url', 'location', 'organization'],
            'conference': ['pages', 'doi', 'url', 'location', 'organization'],
            'incollection': ['pages', 'chapter', 'edition'],
            'phdthesis': ['address', 'month'],
            'mastersthesis': ['address', 'month'],
            'techreport': ['number', 'address', 'month'],
            'misc': ['howpublished', 'url'],
            'unpublished': []
        }
        
        # 不必要的字段（通常应该移除）
        self.unnecessary_fields = ['note', 'keywords', 'abstract', 'file', 
                                  'timestamp', 'owner', 'review', 'rating',
                                  'read', 'printed', 'groups', 'priority']
        
        # 月份映射
        self.month_map = {
            '1': 'jan', '01': 'jan', 'january': 'jan',
            '2': 'feb', '02': 'feb', 'february': 'feb',
            '3': 'mar', '03': 'mar', 'march': 'mar',
            '4': 'apr', '04': 'apr', 'april': 'apr',
            '5': 'may', '05': 'may',
            '6': 'jun', '06': 'jun', 'june': 'jun',
            '7': 'jul', '07': 'jul', 'july': 'jul',
            '8': 'aug', '08': 'aug', 'august': 'aug',
            '9': 'sep', '09': 'sep', 'september': 'sep',
            '10': 'oct', 'october': 'oct',
            '11': 'nov', 'november': 'nov',
            '12': 'dec', 'december': 'dec'
        }
        
        # 学术搜索API配置
        self.crossref_api = "https://api.crossref.org/works"
        self.headers = {
            'User-Agent': 'BibTeXCorrector/1.0 (mailto:user@example.com)'
        }

    def search_crossref_by_doi(self, doi: str) -> Optional[Dict]:
        """
        通过DOI在Crossref查找论文信息
        """
        if not doi:
            return None
            
        # 清理DOI
        doi = self.clean_doi(doi)
        
        try:
            url = f"{self.crossref_api}/{doi}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('message', {})
            
        except Exception as e:
            logger.debug(f"Crossref DOI查询失败: {e}")
        
        return None

    def search_crossref_by_title(self, title: str, author: str = None) -> Optional[Dict]:
        """
        通过标题在Crossref查找论文信息
        """
        if not title:
            return None
            
        # 清理标题
        title_clean = re.sub(r'[{}]', '', title)
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()
        
        try:
            # 构建查询
            query = title_clean
            if author:
                # 提取第一个作者的姓
                first_author = author.split(' and ')[0]
                if ',' in first_author:
                    last_name = first_author.split(',')[0].strip()
                else:
                    last_name = first_author.split()[-1] if first_author.split() else ''
                if last_name:
                    query += f" {last_name}"
            
            params = {
                'query': query,
                'rows': 3  # 获取前3个结果
            }
            
            response = requests.get(self.crossref_api, params=params, 
                                  headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                if items:
                    # 找到最匹配的结果
                    for item in items:
                        item_title = item.get('title', [''])[0].lower()
                        if self.similar_titles(title_clean.lower(), item_title):
                            return item
                    
                    # 如果没有精确匹配，返回第一个结果
                    return items[0]
            
        except Exception as e:
            logger.debug(f"Crossref标题查询失败: {e}")
        
        return None

    def similar_titles(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """
        判断两个标题是否相似
        """
        # 简单的相似度判断
        title1_words = set(re.findall(r'\w+', title1.lower()))
        title2_words = set(re.findall(r'\w+', title2.lower()))
        
        if not title1_words or not title2_words:
            return False
        
        intersection = len(title1_words & title2_words)
        union = len(title1_words | title2_words)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

    def extract_crossref_data(self, crossref_data: Dict) -> Dict:
        """
        从Crossref数据中提取BibTeX字段
        """
        bib_data = {}
        
        # 标题
        if 'title' in crossref_data and crossref_data['title']:
            bib_data['title'] = crossref_data['title'][0]
        
        # 作者
        if 'author' in crossref_data:
            authors = []
            for author in crossref_data['author']:
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    if given:
                        authors.append(f"{family}, {given}")
                    else:
                        authors.append(family)
            if authors:
                bib_data['author'] = ' and '.join(authors)
        
        # 年份
        if 'published-print' in crossref_data:
            date_parts = crossref_data['published-print'].get('date-parts', [[]])
            if date_parts and date_parts[0]:
                bib_data['year'] = str(date_parts[0][0])
                if len(date_parts[0]) > 1:
                    bib_data['month'] = str(date_parts[0][1])
        elif 'published-online' in crossref_data:
            date_parts = crossref_data['published-online'].get('date-parts', [[]])
            if date_parts and date_parts[0]:
                bib_data['year'] = str(date_parts[0][0])
                if len(date_parts[0]) > 1:
                    bib_data['month'] = str(date_parts[0][1])
        
        # 期刊/会议
        if 'container-title' in crossref_data and crossref_data['container-title']:
            container = crossref_data['container-title'][0]
            if crossref_data.get('type') == 'journal-article':
                bib_data['journal'] = container
            else:
                bib_data['booktitle'] = container
        
        # 卷号
        if 'volume' in crossref_data:
            bib_data['volume'] = str(crossref_data['volume'])
        
        # 期号
        if 'issue' in crossref_data:
            bib_data['number'] = str(crossref_data['issue'])
        
        # 页码
        if 'page' in crossref_data:
            bib_data['pages'] = crossref_data['page'].replace('-', '--')
        
        # DOI
        if 'DOI' in crossref_data:
            bib_data['doi'] = crossref_data['DOI']
        
        # URL
        if 'URL' in crossref_data:
            bib_data['url'] = crossref_data['URL']
        
        # 出版社
        if 'publisher' in crossref_data:
            bib_data['publisher'] = crossref_data['publisher']
        
        # ISBN
        if 'ISBN' in crossref_data and crossref_data['ISBN']:
            bib_data['isbn'] = crossref_data['ISBN'][0]
        
        return bib_data

    def search_arxiv(self, title: str, author: str = None) -> Optional[Dict]:
        """
        在arXiv搜索论文信息
        """
        try:
            import feedparser
            
            # 清理标题
            title_clean = re.sub(r'[{}]', '', title)
            title_clean = re.sub(r'\s+', ' ', title_clean).strip()
            
            # 构建查询
            query = f'ti:"{title_clean}"'
            if author:
                first_author = author.split(' and ')[0]
                if ',' in first_author:
                    last_name = first_author.split(',')[0].strip()
                else:
                    last_name = first_author.split()[-1] if first_author.split() else ''
                if last_name:
                    query += f' AND au:{last_name}'
            
            url = f"http://export.arxiv.org/api/query?search_query={quote(query)}&max_results=1"
            
            feed = feedparser.parse(url)
            
            if feed.entries:
                entry = feed.entries[0]
                
                # 提取信息
                bib_data = {}
                
                # 标题
                if 'title' in entry:
                    bib_data['title'] = entry.title.replace('\n', ' ').strip()
                
                # 作者
                if 'authors' in entry:
                    authors = [author['name'] for author in entry.authors]
                    # 转换为 Last, First 格式
                    formatted_authors = []
                    for author in authors:
                        parts = author.split()
                        if len(parts) >= 2:
                            last_name = parts[-1]
                            first_names = ' '.join(parts[:-1])
                            formatted_authors.append(f"{last_name}, {first_names}")
                        else:
                            formatted_authors.append(author)
                    bib_data['author'] = ' and '.join(formatted_authors)
                
                # 年份
                if 'published' in entry:
                    year = entry.published.split('-')[0]
                    bib_data['year'] = year
                
                # arXiv ID
                if 'id' in entry:
                    arxiv_id = entry.id.split('/')[-1]
                    bib_data['eprint'] = arxiv_id
                    bib_data['archiveprefix'] = 'arXiv'
                    bib_data['url'] = entry.id
                
                return bib_data
                
        except ImportError:
            logger.debug("feedparser未安装，跳过arXiv搜索")
        except Exception as e:
            logger.debug(f"arXiv搜索失败: {e}")
        
        return None

    def clean_doi(self, doi: str) -> str:
        """清理DOI格式"""
        if not doi:
            return doi
        
        # 移除常见的DOI URL前缀
        doi_patterns = [
            r'https?://doi\.org/',
            r'https?://dx\.doi\.org/',
            r'doi:',
            r'DOI:',
        ]
        
        for pattern in doi_patterns:
            doi = re.sub(pattern, '', doi, flags=re.IGNORECASE)
        
        return doi.strip()

    def correct_title(self, title: str) -> str:
        """修正标题格式"""
        if not title:
            return title
            
        # 移除多余的空格
        title = ' '.join(title.split())
        
        # 保护大写缩写和专有名词
        acronyms = re.findall(r'\b[A-Z]{2,}\b', title)
        for acronym in acronyms:
            if '{' + acronym + '}' not in title:
                title = title.replace(acronym, '{' + acronym + '}')
        
        # 规范化引号
        title = re.sub(r'[""]', '"', title)
        title = re.sub(r"['']", "'", title)
        
        return title

    def correct_author(self, author: str) -> str:
        """修正作者格式"""
        if not author:
            return author
            
        # 分割多个作者
        authors = re.split(r'\s+and\s+', author)
        corrected_authors = []
        
        for auth in authors:
            auth = auth.strip()
            if not auth:
                continue
                
            # 如果已经是 "Last, First" 格式
            if ',' in auth:
                corrected_authors.append(auth)
            else:
                # 尝试转换为 "Last, First" 格式
                parts = auth.split()
                if len(parts) >= 2:
                    last_name = parts[-1]
                    first_names = ' '.join(parts[:-1])
                    corrected_authors.append(f"{last_name}, {first_names}")
                else:
                    corrected_authors.append(auth)
        
        return ' and '.join(corrected_authors)

    def correct_pages(self, pages: str) -> str:
        """修正页码格式"""
        if not pages:
            return pages
        
        # 替换各种分隔符为 --
        pages = re.sub(r'\s*[-–—]\s*', '--', pages)
        pages = re.sub(r'\s+', '', pages)
        
        return pages

    def correct_year(self, year: str) -> str:
        """修正年份格式"""
        if not year:
            return year
        
        # 提取4位数字作为年份
        match = re.search(r'\b((19|20)\d{2})\b', year)
        if match:
            return match.group(1)
        
        return year

    def correct_month(self, month: str) -> str:
        """修正月份格式"""
        if not month:
            return month
        
        month_lower = month.lower().strip()
        return self.month_map.get(month_lower, month)

    def enhance_entry_online(self, entry: Dict) -> Dict:
        """
        通过在线搜索增强条目信息
        """
        logger.info(f"  正在搜索条目 '{entry.get('ID', 'unknown')}' 的在线信息...")
        
        online_data = {}
        
        # 首先尝试通过DOI搜索
        if 'doi' in entry:
            logger.info(f"    通过DOI搜索: {entry['doi']}")
            crossref_data = self.search_crossref_by_doi(entry['doi'])
            if crossref_data:
                online_data = self.extract_crossref_data(crossref_data)
                logger.info("    ✓ 找到Crossref信息")
        
        # 如果DOI搜索失败或没有DOI，尝试通过标题搜索
        if not online_data and 'title' in entry:
            logger.info(f"    通过标题搜索...")
            
            # 尝试Crossref
            crossref_data = self.search_crossref_by_title(
                entry.get('title', ''), 
                entry.get('author', '')
            )
            if crossref_data:
                online_data = self.extract_crossref_data(crossref_data)
                logger.info("    ✓ 找到Crossref信息")
            
            # 如果Crossref没找到，尝试arXiv
            if not online_data:
                arxiv_data = self.search_arxiv(
                    entry.get('title', ''),
                    entry.get('author', '')
                )
                if arxiv_data:
                    online_data = arxiv_data
                    logger.info("    ✓ 找到arXiv信息")
        
        # 合并在线数据到条目
        if online_data:
            # 更新标题（使用在线找到的准确标题）
            if 'title' in online_data:
                old_title = entry.get('title', '')
                entry['title'] = self.correct_title(online_data['title'])
                if old_title and old_title != entry['title']:
                    logger.info(f"    标题已更正")
            
            # 补充缺失的字段
            for field, value in online_data.items():
                if field not in entry or not entry[field]:
                    entry[field] = value
                    logger.info(f"    补充字段: {field}")
            
            # 更新不准确的字段（除了ID）
            for field in ['author', 'year', 'pages', 'volume', 'number', 'doi']:
                if field in online_data and field in entry:
                    if field == 'author':
                        # 作者可能格式不同，需要特殊处理
                        entry[field] = self.correct_author(online_data[field])
                    elif entry[field] != online_data[field]:
                        entry[field] = online_data[field]
                        logger.info(f"    更新字段: {field}")
        else:
            logger.info("    未找到在线信息")
        
        return entry

    def remove_unnecessary_fields(self, entry: Dict) -> Dict:
        """
        移除不必要的字段
        """
        entry_type = entry.get('ENTRYTYPE', 'misc').lower()
        
        # 获取必需和可选字段
        required = set(self.required_fields.get(entry_type, []))
        optional = set(self.optional_fields.get(entry_type, []))
        
        # 保留的字段：ID, ENTRYTYPE, 必需字段，可选字段
        keep_fields = {'ID', 'ENTRYTYPE'} | required | optional
        
        # 额外保留一些通用有用的字段
        keep_fields |= {'doi', 'url', 'eprint', 'archiveprefix'}
        
        # 移除不在保留列表中的字段
        fields_to_remove = []
        for field in entry:
            if field.lower() not in keep_fields and field not in keep_fields:
                fields_to_remove.append(field)
        
        for field in fields_to_remove:
            if field.lower() in self.unnecessary_fields:
                logger.info(f"    移除不必要字段: {field}")
                del entry[field]
        
        return entry

    def correct_entry(self, entry: Dict) -> Dict:
        """
        修正单个BibTeX条目
        """
        logger.info(f"\n处理条目: {entry.get('ID', 'unknown')}")
        
        # 复制条目以避免修改原始数据
        corrected = entry.copy()
        
        # 保持原始ID不变
        original_id = corrected.get('ID', '')
        
        # 在线搜索增强（如果启用）
        if self.search_online:
            corrected = self.enhance_entry_online(corrected)
            time.sleep(0.5)  # 避免请求过快
        
        # 基础格式修正
        if 'title' in corrected:
            corrected['title'] = self.correct_title(corrected['title'])
        
        if 'author' in corrected:
            corrected['author'] = self.correct_author(corrected['author'])
        
        if 'pages' in corrected:
            corrected['pages'] = self.correct_pages(corrected['pages'])
        
        if 'doi' in corrected:
            corrected['doi'] = self.clean_doi(corrected['doi'])
        
        if 'year' in corrected:
            corrected['year'] = self.correct_year(corrected['year'])
        
        if 'month' in corrected:
            corrected['month'] = self.correct_month(corrected['month'])
        
        # 移除不必要的字段
        if self.remove_unnecessary:
            corrected = self.remove_unnecessary_fields(corrected)
        
        # 移除空字段
        corrected = {k: v for k, v in corrected.items() if v and str(v).strip()}
        
        # 恢复原始ID（不更改）
        if original_id:
            corrected['ID'] = original_id
        
        # 检查必需字段
        entry_type = corrected.get('ENTRYTYPE', 'misc').lower()
        if entry_type in self.required_fields:
            missing_fields = []
            for field in self.required_fields[entry_type]:
                if field not in corrected:
                    missing_fields.append(field)
            
            if missing_fields:
                logger.warning(f"  警告: 缺少必需字段: {', '.join(missing_fields)}")
        
        return corrected

    def correct_bibtex_file(self, input_file: str, output_file: Optional[str] = None) -> None:
        """
        修正整个BibTeX文件
        """
        # 读取BibTeX文件
        with open(input_file, 'r', encoding='utf-8') as bibtex_file:
            parser = bibtexparser.bparser.BibTexParser(common_strings=True)
            parser.ignore_nonstandard_types = False
            parser.homogenize_fields = True
            bib_database = bibtexparser.load(bibtex_file, parser)
        
        logger.info(f"\n读取到 {len(bib_database.entries)} 个条目")
        logger.info("=" * 60)
        
        # 修正每个条目
        corrected_entries = []
        for i, entry in enumerate(bib_database.entries, 1):
            logger.info(f"[{i}/{len(bib_database.entries)}]")
            corrected_entry = self.correct_entry(entry)
            corrected_entries.append(corrected_entry)
        
        # 创建新的数据库
        new_db = BibDatabase()
        new_db.entries = corrected_entries
        
        # 设置输出格式
        writer = BibTexWriter()
        writer.indent = '    '
        writer.order_entries_by = None  # 保持原始顺序
        writer.add_trailing_commas = True
        writer.align_values = True
        writer.common_strings = True
        
        # 输出文件
        if output_file is None:
            output_file = input_file.replace('.bib', '_corrected.bib')
        
        with open(output_file, 'w', encoding='utf-8') as bibtex_output:
            bibtex_output.write(writer.write(new_db))
        
        logger.info(f"\n\n修正完成！")
        logger.info(f"输出文件: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='高级BibTeX条目修正工具 - 自动查找缺失信息并修正格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能特点:
  - 自动从Crossref和arXiv查找准确的论文信息
  - 更正标题为最新版本
  - 补全缺失的页码、卷号、DOI等信息
  - 移除不必要的字段（如note, keywords等）
  - 保持原始条目ID不变
  - 统一格式规范

示例:
  %(prog)s input.bib                    # 修正input.bib
  %(prog)s input.bib -o output.bib      # 指定输出文件
  %(prog)s input.bib --offline          # 仅离线修正格式
  %(prog)s input.bib --keep-all-fields  # 保留所有字段
        """
    )
    
    parser.add_argument('input', help='输入的BibTeX文件路径')
    parser.add_argument('-o', '--output', help='输出的BibTeX文件路径')
    parser.add_argument('--offline', action='store_true', 
                       help='仅离线修正，不进行网络搜索')
    parser.add_argument('--keep-all-fields', action='store_true',
                       help='保留所有字段，不移除不必要的字段')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    corrector = AdvancedBibCorrector(
        search_online=not args.offline,
        remove_unnecessary=not args.keep_all_fields
    )
    
    try:
        corrector.correct_bibtex_file(args.input, args.output)
    except FileNotFoundError:
        logger.error(f"错误: 找不到文件 '{args.input}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
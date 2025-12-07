#!/usr/bin/env python3
"""
BibTeX条目修正工具
用于规范化和修正BibTeX条目的格式，使其更加严谨和一致
"""

import re
import sys
from typing import Dict, List, Optional
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.customization import *
import argparse


class BibCorrector:
    """BibTeX条目修正器"""
    
    def __init__(self):
        """初始化修正器"""
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
            'unpublished': ['author', 'title', 'note']
        }
        
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

    def correct_title(self, title: str) -> str:
        """
        修正标题格式
        - 保护专有名词和缩写的大小写
        - 规范化空格和特殊字符
        """
        if not title:
            return title
            
        # 移除多余的空格
        title = ' '.join(title.split())
        
        # 保护大写缩写和专有名词（用花括号括起来）
        # 查找所有连续的大写字母（长度>=2）或已知的专有名词
        acronyms = re.findall(r'\b[A-Z]{2,}\b', title)
        for acronym in acronyms:
            title = title.replace(acronym, '{' + acronym + '}')
        
        # 保护已有的花括号内容
        protected = re.findall(r'\{[^}]+\}', title)
        
        # 规范化引号
        title = re.sub(r'[""]', '"', title)
        title = re.sub(r"['']", "'", title)
        
        return title

    def correct_author(self, author: str) -> str:
        """
        修正作者格式
        - 统一为 "Last, First" 或 "Last, First and Last, First" 格式
        - 处理特殊字符
        """
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
                    # 假设最后一个是姓，前面的是名
                    last_name = parts[-1]
                    first_names = ' '.join(parts[:-1])
                    corrected_authors.append(f"{last_name}, {first_names}")
                else:
                    corrected_authors.append(auth)
        
        return ' and '.join(corrected_authors)

    def correct_pages(self, pages: str) -> str:
        """
        修正页码格式
        - 统一使用 -- 作为页码范围分隔符
        """
        if not pages:
            return pages
            
        # 替换各种分隔符为 --
        pages = re.sub(r'\s*[-–—]\s*', '--', pages)
        pages = re.sub(r'\s+', '', pages)  # 移除空格
        
        return pages

    def correct_doi(self, doi: str) -> str:
        """
        修正DOI格式
        - 移除URL前缀，只保留DOI
        """
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

    def correct_year(self, year: str) -> str:
        """
        修正年份格式
        - 确保是4位数字
        """
        if not year:
            return year
            
        # 提取4位数字作为年份
        match = re.search(r'\b((19|20)\d{2})\b', year)
        if match:
            return match.group(1)
        
        return year

    def correct_month(self, month: str) -> str:
        """
        修正月份格式
        - 统一为三字母缩写
        """
        if not month:
            return month
            
        month_lower = month.lower().strip()
        return self.month_map.get(month_lower, month)

    def correct_url(self, url: str) -> str:
        """
        修正URL格式
        - 确保URL格式正确
        """
        if not url:
            return url
            
        url = url.strip()
        
        # 如果没有协议，添加 https://
        if not re.match(r'^https?://', url):
            url = 'https://' + url
        
        return url

    def generate_id(self, entry: Dict) -> str:
        """
        生成标准的条目ID
        格式: AuthorYear 或 AuthorYearKeyword
        """
        # 获取第一作者的姓
        if 'author' in entry:
            authors = re.split(r'\s+and\s+', entry['author'])
            if authors:
                first_author = authors[0]
                # 提取姓（逗号前的部分或最后一个单词）
                if ',' in first_author:
                    last_name = first_author.split(',')[0]
                else:
                    last_name = first_author.split()[-1] if first_author.split() else 'Unknown'
                last_name = re.sub(r'[^a-zA-Z]', '', last_name)
            else:
                last_name = 'Unknown'
        else:
            last_name = 'Unknown'
        
        # 获取年份
        year = entry.get('year', 'YYYY')
        if len(year) > 4:
            year = year[:4]
        
        # 获取标题的第一个有意义的单词
        if 'title' in entry:
            title_words = re.findall(r'\b[a-zA-Z]+\b', entry['title'].lower())
            # 跳过常见的冠词和介词
            stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
            keywords = [w for w in title_words if w not in stop_words]
            keyword = keywords[0].capitalize() if keywords else ''
        else:
            keyword = ''
        
        # 生成ID
        if keyword:
            new_id = f"{last_name}{year}{keyword}"
        else:
            new_id = f"{last_name}{year}"
        
        return new_id

    def correct_entry(self, entry: Dict) -> Dict:
        """
        修正单个BibTeX条目
        """
        # 复制条目以避免修改原始数据
        corrected = entry.copy()
        
        # 修正各个字段
        if 'title' in corrected:
            corrected['title'] = self.correct_title(corrected['title'])
        
        if 'author' in corrected:
            corrected['author'] = self.correct_author(corrected['author'])
        
        if 'pages' in corrected:
            corrected['pages'] = self.correct_pages(corrected['pages'])
        
        if 'doi' in corrected:
            corrected['doi'] = self.correct_doi(corrected['doi'])
        
        if 'year' in corrected:
            corrected['year'] = self.correct_year(corrected['year'])
        
        if 'month' in corrected:
            corrected['month'] = self.correct_month(corrected['month'])
        
        if 'url' in corrected:
            corrected['url'] = self.correct_url(corrected['url'])
        
        # 移除空字段
        corrected = {k: v for k, v in corrected.items() if v and v.strip()}
        
        # 检查必需字段
        entry_type = corrected.get('ENTRYTYPE', 'misc').lower()
        if entry_type in self.required_fields:
            missing_fields = []
            for field in self.required_fields[entry_type]:
                if field not in corrected:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"警告: 条目 '{corrected.get('ID', 'unknown')}' 缺少必需字段: {', '.join(missing_fields)}")
        
        # 生成标准ID（可选）
        if 'ID' in corrected:
            new_id = self.generate_id(corrected)
            if new_id != corrected['ID']:
                print(f"建议: 将ID '{corrected['ID']}' 改为 '{new_id}'")
                # 可以选择是否自动更改ID
                # corrected['ID'] = new_id
        
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
        
        print(f"读取到 {len(bib_database.entries)} 个条目")
        
        # 修正每个条目
        corrected_entries = []
        for entry in bib_database.entries:
            corrected_entry = self.correct_entry(entry)
            corrected_entries.append(corrected_entry)
        
        # 创建新的数据库
        new_db = BibDatabase()
        new_db.entries = corrected_entries
        
        # 设置输出格式
        writer = BibTexWriter()
        writer.indent = '    '  # 使用4个空格缩进
        writer.order_entries_by = None  # 保持原始顺序
        writer.add_trailing_commas = True  # 在最后一个字段后添加逗号
        writer.align_values = True  # 对齐值
        writer.common_strings = True  # 使用通用字符串（如月份缩写）
        
        # 输出文件
        if output_file is None:
            output_file = input_file.replace('.bib', '_corrected.bib')
        
        with open(output_file, 'w', encoding='utf-8') as bibtex_output:
            bibtex_output.write(writer.write(new_db))
        
        print(f"修正后的BibTeX已保存到: {output_file}")

    def correct_bibtex_string(self, bib_string: str) -> str:
        """
        修正BibTeX字符串（用于单个条目或小段BibTeX）
        """
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        parser.homogenize_fields = True
        
        bib_database = bibtexparser.loads(bib_string, parser)
        
        # 修正条目
        corrected_entries = []
        for entry in bib_database.entries:
            corrected_entry = self.correct_entry(entry)
            corrected_entries.append(corrected_entry)
        
        # 创建新的数据库
        new_db = BibDatabase()
        new_db.entries = corrected_entries
        
        # 设置输出格式
        writer = BibTexWriter()
        writer.indent = '    '
        writer.order_entries_by = None
        writer.add_trailing_commas = True
        writer.align_values = True
        writer.common_strings = True
        
        return writer.write(new_db)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='修正和规范化BibTeX条目格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s input.bib                    # 修正input.bib，输出到input_corrected.bib
  %(prog)s input.bib -o output.bib      # 修正input.bib，输出到output.bib
  %(prog)s -s "@article{...}"           # 修正BibTeX字符串并输出到控制台
        """
    )
    
    parser.add_argument('input', nargs='?', help='输入的BibTeX文件路径')
    parser.add_argument('-o', '--output', help='输出的BibTeX文件路径')
    parser.add_argument('-s', '--string', help='直接输入BibTeX字符串进行修正')
    
    args = parser.parse_args()
    
    corrector = BibCorrector()
    
    if args.string:
        # 处理字符串输入
        corrected = corrector.correct_bibtex_string(args.string)
        print("修正后的BibTeX:")
        print(corrected)
    elif args.input:
        # 处理文件输入
        corrector.correct_bibtex_file(args.input, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
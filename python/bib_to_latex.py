import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding

# Dictionary to map venue keywords to abbreviations (CCF Recommended Conferences and more specific examples)
VENUE_ABBREVIATIONS = {
    'COLM' : 'COLM',
    'Transactions on Machine Learning Research' : 'TMLR',
    'International Conference on Machine Learning' : 'ICML',
    'COLING': 'COLING',
    'ACM SIGKDD': 'KDD',
    'IEEE/CVF Conference on Computer Vision and Pattern Recognition': 'CVPR',
    'arXiv preprint': 'Preprint',
    'International Conference on Learning Representations': 'ICLR',
    'Advances in Neural Information Processing Systems': 'NeurIPS',
    'North American Chapter of the Association for Computational Linguistics': 'NAACL',
    'NAACL' : 'NAACL',
    'Association for Computational Linguistics': 'ACL',
    'International Conference on Data Mining': 'ICDM',
    'International Joint Conference on Artificial Intelligence': 'IJCAI',
    'Conference on Computer Vision and Pattern Recognition': 'CVPR',
    'International Conference on Computer Vision': 'ICCV',
    'International Conference on Database Systems for Advanced Applications': 'DASFAA',
    'Symposium on Theory of Computing': 'STOC',
    'International Symposium on Computer Architecture': 'ISCA',
    'ACM Conference on Information and Knowledge Management': 'CIKM',
    'International Conference on Web Search and Data Mining': 'WSDM',
    'International Conference on Very Large Data Bases': 'VLDB',
    'Annual Meeting of the Association for Computational Linguistics': 'ACL',
    'International Conference on Automated Planning and Scheduling': 'ICAPS',
    'International Symposium on Information Theory': 'ISIT',
    'IEEE International Conference on Robotics and Automation': 'ICRA',
    'European Conference on Computer Vision': 'ECCV',
    'International Symposium on Software Testing and Analysis': 'ISSTA',
    'International Symposium on Theoretical Aspects of Software Engineering': 'TASE',
    'International Conference on Artificial Intelligence and Statistics': 'AISTATS',
    'IEEE International Conference on Data Engineering': 'ICDE',
    'International Conference on Principles of Knowledge Representation and Reasoning': 'KR',
    'Conference on Uncertainty in Artificial Intelligence': 'UAI',
    'International Conference on Knowledge Capture': 'K-CAP',
    'IEEE International Conference on Computer Communications': 'INFOCOM',
    'IEEE Transactions on Signal Processing': 'TSP',
    'Proceedings of the 2017 ACM on Conference on Information and Knowledge Management': 'CIKM',
    'Proceedings of the Web Conference': 'WWW',
    'International Symposium on Software Reliability Engineering': 'ISSRE',
    'ECCV': 'ECCV',
    'Machine Learning and Knowledge Discovery in Databases: European': 'ECML',
    'Learning on Graph': 'LOG',
    'AAAI': 'AAAI',
    'Transactions on Knowledge and Data Engineering': 'TKDE',
    'European Conference on Machine Learning': 'ECML',
    'CONLL': 'CONLL',
    'EMNLP': 'EMNLP',
    'Empirical Methods': 'EMNLP',
    # Add more mappings for common conferences/journals
}

# CCF Ratings for conferences and journals
CCF_RATINGS = {
    'ICML': 'A',
    'KDD': 'A',
    'CVPR': 'A',
    'NeurIPS': 'A',
    'ACL': 'A',
    'IJCAI': 'A',
    'ICCV': 'A',
    'STOC': 'A',
    'ISCA': 'A',
    'VLDB': 'A',
    'ECCV': 'A',
    'ICDE': 'A',
    'WWW': 'A',
    'AAAI': 'A',
    
    'CIKM': 'B',
    'WSDM': 'B',
    'ICAPS': 'B',
    'ISIT': 'B',
    'ICRA': 'B',
    'AISTATS': 'B',
    'UAI': 'B',
    'INFOCOM': 'B',
    'COLING': 'B',
    'EMNLP': 'B',
    'NAACL': 'B',
    'ECML': 'B',
    
    'DASFAA': 'C',
    'ISSTA': 'C',
    'TASE': 'C',
    'K-CAP': 'C',
    'ISSRE': 'C',
    'ICDM': 'C',
    'CONLL': 'C',
    
    'Preprint': 'N/A',
    'LOG': 'N/A',
    'TKDE': 'A',  # IEEE Transactions on Knowledge and Data Engineering
    'TSP': 'A',   # IEEE Transactions on Signal Processing
    # Add more CCF ratings as needed
}

# Colors for different CCF ratings
CCF_COLORS = {
    'A': 'red',
    'B': 'blue',
    'C': 'green',
    'N/A': 'black'
}

# 定义会议时间顺序
CONFERENCE_TIME_ORDER = [
    'TMLR',
    'NeurIPS',
    'EMNLP',
    'COLM',
    'ACL',
    'ICML',
    'CVPR',
    'ICLR',
    'NAACL',
    'ICCV',
    'AAAI'
    # 你可以根据需要添加更多的会议
]

# 函数：根据缩写表替换venue
def abbreviate_venue(venue):
    for key, abbreviation in VENUE_ABBREVIATIONS.items():
        if key in venue:
            return abbreviation
    return venue

# 函数：将作者名字从 "Last Name, First Name" 改为 "First Name Last Name"
def format_author_names(authors_str):
    authors = authors_str.split(' and ')
    formatted_authors = []
    for author in authors:
        parts = author.split(', ')
        if len(parts) == 2:
            # 改为 "First Name Last Name" 格式
            formatted_authors.append(f"{parts[1]} {parts[0]}")
        else:
            formatted_authors.append(author)
    return formatted_authors

# 函数：获取会议在排序中的位置
def get_conference_order(venue):
    if venue in CONFERENCE_TIME_ORDER:
        return CONFERENCE_TIME_ORDER.index(venue)
    else:
        # 如果会议不在列表中，返回一个较大的值以确保排在后面
        return len(CONFERENCE_TIME_ORDER)

def bib_to_paper_list(bib_file):
    # 读取.bib文件并使用BibTexParser
    with open(bib_file) as bibtex_file:
        parser = BibTexParser(common_strings=True)  # 保留特殊字符并禁用多余的转义
        bib_database = bibtexparser.load(bibtex_file, parser=parser)
    
    paper_list = []

    for entry in bib_database.entries:
        # 跳过arXiv论文
        if 'arxiv' in entry.get('journal', '').lower():
            continue

        authors_str = entry.get('author', 'Unknown')
        authors = format_author_names(authors_str)  # 改为 "First Name Last Name" 格式
        title = entry.get('title', 'Title not available').rstrip('.')
        year = entry.get('year', '')
        venue = entry.get('booktitle', entry.get('journal', ''))
        venue = abbreviate_venue(venue)  # 使用缩写规则

        # 生成格式化条目，去除页码，保持特殊字符和LaTeX格式
        formatted_entry = f"{authors[0]}, {', '.join(authors[1:])}. {title}, \\textit{{{venue} {year}}}.".strip()

        # 存储年份、venue和条目，用于排序
        paper_list.append((int(year), venue, formatted_entry))

    # 按年份降序排序，然后按会议时间顺序进行排序
    paper_list.sort(key=lambda x: (-x[0], get_conference_order(x[1]), x[1].lower() if x[1] else ''))

    # 为每篇论文添加编号 [1], [2], 等
    numbered_papers = [f"[{i+1}] {entry}" for i, (_, _, entry) in enumerate(paper_list)]
    
    # 返回带有编号的论文列表
    return "\n\n".join(numbered_papers)

# 新函数：带有CCF评级的论文列表（仅显示A类会议/期刊）
def bib_to_paper_list_ccf(bib_file):
    # 读取.bib文件并使用BibTexParser
    with open(bib_file) as bibtex_file:
        parser = BibTexParser(common_strings=True)  # 保留特殊字符并禁用多余的转义
        bib_database = bibtexparser.load(bibtex_file, parser=parser)
    
    paper_list = []

    for entry in bib_database.entries:
        # 跳过arXiv论文
        if 'arxiv' in entry.get('journal', '').lower():
            continue

        authors_str = entry.get('author', 'Unknown')
        authors = format_author_names(authors_str)  # 改为 "First Name Last Name" 格式
        title = entry.get('title', 'Title not available').rstrip('.')
        year = entry.get('year', '')
        venue = entry.get('booktitle', entry.get('journal', ''))
        venue_abbr = abbreviate_venue(venue)  # 使用缩写规则
        
        # 获取CCF评级
        ccf_rating = CCF_RATINGS.get(venue_abbr, '')
        
        # 仅为A类会议/期刊添加CCF标签，其他不添加
        if ccf_rating == 'A':
            # 使用字符串作为颜色名称
            ccf_label = f"\\textcolor{{red}}{{(CCF-A)}}"
            formatted_entry = f"{authors[0]}, {', '.join(authors[1:])}. {title}, \\textit{{{venue_abbr} {year}}}. {ccf_label}".strip()
        else:
            formatted_entry = f"{authors[0]}, {', '.join(authors[1:])}. {title}, \\textit{{{venue_abbr} {year}}}.".strip()

        # 存储年份、venue和条目，用于排序
        paper_list.append((int(year), venue_abbr, formatted_entry))

    # 按年份降序排序，然后按会议时间顺序进行排序
    paper_list.sort(key=lambda x: (-x[0], get_conference_order(x[1]), x[1].lower() if x[1] else ''))

    # 为每篇论文添加编号 [1], [2], 等
    numbered_papers = [f"[{i+1}] {entry}" for i, (_, _, entry) in enumerate(paper_list)]
    
    # 返回带有编号的论文列表
    return "\n\n".join(numbered_papers)

# 示例用法
bib_file = "../citations.bib"
output_file = "paper_list.tex"  # 生成的 LaTeX 文件
output_file_ccf = "paper_list_ccf.tex"  # 生成的带CCF评级的 LaTeX 文件

# 将结果写入到 LaTeX 文件
with open(output_file, "w") as f:
    f.write(bib_to_paper_list(bib_file))

# 将带CCF评级的结果写入到 LaTeX 文件
with open(output_file_ccf, "w") as f:
    f.write(bib_to_paper_list_ccf(bib_file))

print('done')
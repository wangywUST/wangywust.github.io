import os
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding

# Dictionary to map venue keywords to abbreviations (CCF Recommended Conferences and more specific examples)
VENUE_ABBREVIATIONS = {
    'ACM SIGKDD': 'KDD',
    'IEEE/CVF Conference on Computer Vision and Pattern Recognition': 'CVPR',
    'arXiv preprint': 'Preprint',
    'International Conference on Learning Representations': 'ICLR',
    'Advances in Neural Information Processing Systems': 'NeurIPS',
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
    # Add more mappings for common conferences/journals
}

# Function to convert 'Last, First' to 'First Last'
def convert_author_format(author_name):
    if ',' in author_name:
        parts = author_name.split(',')
        return f"{parts[1].strip()} {parts[0].strip()}"
    return author_name  # If no comma, return as is

# Function to clean and abbreviate the venue for the image badge using keyword matching
def abbreviate_venue(venue):
    # Clean venue by removing backslashes and special characters
    venue_cleaned = venue.replace("\\", "").replace("&", "and")
    
    # Match based on keywords in the venue name
    for keyword, short_name in VENUE_ABBREVIATIONS.items():
        if keyword in venue_cleaned:
            return short_name
    return venue_cleaned  # Return cleaned venue if no abbreviation is found

# Function to clean venue for full name (without abbreviating)
def clean_full_venue(venue):
    # Remove special characters like backslashes and replace '&' with 'and'
    return venue.replace("\\", "").replace("&", "and")

# Function to clean and remove special characters from title, including replacing \textquotesingle and \ast
def clean_title(title):
    return title.replace("{", "").replace("}", "").replace("\\textquotesingle", "\u0027").replace("\\ast", "*")

# Function to generate image path based on the bib entry
def generate_image_filename(entry):
    # Get the entry ID, which is typically the bibtex citation key
    entry_id = entry.get('ID', 'default')
    return f"Image/{entry_id}.png"

# Function to generate HTML from bib entry, without the image if the file doesn't exist
def generate_html(entry):
    # Handling multiple authors by splitting and formatting them
    authors = entry.get('author', 'Unknown Author').split(" and ")
    formatted_authors = ', '.join([convert_author_format(author) for author in authors])

    # Clean the title to remove special characters
    clean_title_text = clean_title(entry.get('title', 'Untitled'))

    # Get the full venue (journal or booktitle) and its abbreviation for the image badge
    full_venue = entry.get('journal', entry.get('booktitle', 'Preprint'))
    abbreviated_venue = abbreviate_venue(full_venue)
    cleaned_full_venue = clean_full_venue(full_venue)  # Cleaned full venue without abbreviations

    # Use the entry's ID to generate the image filename
    img_src = generate_image_filename(entry)
    
    # Check if the image file exists
    image_exists = os.path.exists(img_src)
    
    # If the image exists, include the image and layout as before
    if image_exists:
        html = f'''
<li>
<div class="pub-row">
  <div class="col-sm-3 abbr" style="position: relative;padding-right: 15px;padding-left: 15px;">
    <img src="{img_src}" class="teaser img-fluid z-depth-1">
            <abbr class="badge">{abbreviated_venue}</abbr>
  </div>
  <div class="col-sm-9" style="position: relative;padding-right: 15px;padding-left: 20px;">
      <div class="title"><a href="{entry.get('url', '#')}">{clean_title_text}</a></div>
      <div class="author"><strong>{formatted_authors}</strong>.</div>
      <div class="periodical"><em>{cleaned_full_venue}, {entry.get('year', '2024')}.</em></div>
      <div class="links">
        <a href="{entry.get('url', '#')}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">PDF</a>
        <a href="{entry.get('code', '#')}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">Code</a>
      </div>
  </div>
</div>
</li>
'''
    else:
        # If the image doesn't exist, remove the image section and extend the text
        html = f'''
<li>
<div class="pub-row">
  <div class="col-sm-12" style="position: relative;padding-right: 15px;padding-left: 15px;">
      <div class="title"><a href="{entry.get('url', '#')}">{clean_title_text}</a></div>
      <div class="author"><strong>{formatted_authors}</strong>.</div>
      <div class="periodical"><em>{cleaned_full_venue}, {entry.get('year', '2024')}.</em></div>
      <div class="links">
        <a href="{entry.get('url', '#')}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">PDF</a>
        <a href="{entry.get('code', '#')}" class="btn btn-sm z-depth-0" role="button" target="_blank" style="font-size:12px;">Code</a>
      </div>
  </div>
</div>
</li>
'''

    return html

# Load the .bib file
def load_bib_file(file_path):
    with open(file_path, 'r') as bib_file:
        bib_database = bibtexparser.load(bib_file, parser=BibTexParser(customization=homogenize_latex_encoding))
    return bib_database.entries

# Sort entries by year in descending order
def sort_entries_by_year(entries):
    return sorted(entries, key=lambda x: int(x.get('year', 0)), reverse=True)

# Generate HTML for all entries in the bib file, adding a divider for each year
def generate_bibliography_html(entries):
    # Sort entries by year from newest to oldest
    sorted_entries = sort_entries_by_year(entries)

    html_content = '''<h2 id="publications" style="margin: 2px 0px -15px;">Selected Publications <temp style="font-size:15px;">[</temp><a href="https://scholar.google.com/citations?user=Sh9QvBkAAAAJ&hl=en" target="_blank" style="font-size:15px;">Google Scholar</a><temp style="font-size:15px;">]</temp><temp style="font-size:15px;">[</temp><a href="https://dblp.org/pid/50/5889-1.html" target="_blank" style="font-size:15px;">DBLP</a><temp style="font-size:15px;">]</temp></h2>

<div class="publications">
<ol class="bibliography">
'''
    last_year = None
    for entry in sorted_entries:
        current_year = entry.get('year', 'Unknown')
        if current_year != last_year:
            # Add a divider for each new year
            html_content += f'<h3>{current_year}</h3><hr>'
            last_year = current_year
        html_content += generate_html(entry)
    html_content += '</ol>\n</div>'
    return html_content

# Main function to execute the conversion
def convert_bib_to_html(bib_file_path, output_html_path):
    entries = load_bib_file(bib_file_path)
    html_content = generate_bibliography_html(entries)
    
    # Write the HTML to the output file
    with open(output_html_path, 'w') as html_file:
        html_file.write(html_content)
    
    print(f"HTML file successfully written to {output_html_path}")

# Example usage:
convert_bib_to_html('citations.bib', '_includes/output_file.md')

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding

# Function to convert 'Last, First' to 'First Last'
def convert_author_format(author_name):
    if ',' in author_name:
        parts = author_name.split(',')
        return f"{parts[1].strip()} {parts[0].strip()}"
    return author_name  # If no comma, return as is

# Function to remove special characters from venue
def clean_venue(venue):
    return venue.replace("\\", "")  # Remove backslashes

# Function to generate HTML from bib entry
def generate_html(entry):
    # Handling multiple authors by splitting and formatting them
    authors = entry.get('author', 'Unknown Author').split(" and ")
    formatted_authors = ', '.join([convert_author_format(author) for author in authors])

    # Clean the venue (journal or booktitle) to remove special characters
    venue = clean_venue(entry.get('journal', entry.get('booktitle', 'Preprint')))

    # Assuming you want to use the entry's 'ID' as part of the image filename
    img_src = f"Image/{entry.get('ID', 'default')}.png"
    
    html = f'''
<li>
<div class="pub-row">
  <div class="col-sm-3 abbr" style="position: relative;padding-right: 15px;padding-left: 15px;">
    <img src="{img_src}" class="teaser img-fluid z-depth-1">
            <abbr class="badge">{venue}</abbr>
  </div>
  <div class="col-sm-9" style="position: relative;padding-right: 15px;padding-left: 20px;">
      <div class="title"><a href="{entry.get('url', '#')}">{entry.get('title', 'Untitled')}</a></div>
      <div class="author"><strong>{formatted_authors}</strong>.</div>
      <div class="periodical"><em>{venue}, {entry.get('year', '2024')}.</em></div>
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

# Generate HTML for all entries in the bib file
def generate_bibliography_html(entries):
    html_content = '''<h2 id="publications" style="margin: 2px 0px -15px;">Selected Publications <temp style="font-size:15px;">[</temp><a href="https://scholar.google.com/citations?user=Sh9QvBkAAAAJ&hl=en" target="_blank" style="font-size:15px;">Google Scholar</a><temp style="font-size:15px;">]</temp><temp style="font-size:15px;">[</temp><a href="https://dblp.org/pid/50/5889-1.html" target="_blank" style="font-size:15px;">DBLP</a><temp style="font-size:15px;">]</temp></h2>

<div class="publications">
<ol class="bibliography">
'''
    for entry in entries:
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

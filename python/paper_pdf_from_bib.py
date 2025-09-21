#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_pdf_from_bib.py
---------------------
Given a .bib file, try to download PDFs for each BibTeX entry and name them according to a filename template.

Data sources (in order):
  1) Semantic Scholar (openAccessPdf or arXiv via externalIds; optional DOIâ†’Unpaywall)
  2) arXiv API
  3) OpenAlex
  4) Unpaywall (optional, requires env UNPAYWALL_EMAIL)

Outputs:
  - Downloads PDFs into --outdir (default: ./pdfs)
  - Writes a manifest CSV (paper_pdf_manifest.csv) with status for each entry
  - Optional: dry-run mode to preview actions

Usage:
  python paper_pdf_from_bib.py --bib refs.bib --outdir ./pdfs
  python paper_pdf_from_bib.py --bib refs.bib --outdir ./pdfs --template "{citekey} - {first_author}{year} - {title}"
  python paper_pdf_from_bib.py --bib refs.bib --outdir ./pdfs --dry-run

Notes:
  - Requires 'requests' package (pip install requests)
  - Set UNPAYWALL_EMAIL environment variable to enable Unpaywall lookups.
  - Internet access is required when actually running the script on your machine.
"""

import os
import re
import csv
import json
import time
import html
import argparse
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher

try:
    import requests
except Exception:
    print("This script requires the 'requests' package. Install with: pip install requests")
    raise

UA = {"User-Agent": "pdf-finder/1.0 (paper_pdf_from_bib.py)"}

# -------------------- Utilities --------------------

def norm_title(t: str) -> str:
    t = t or ""
    t = t.lower().strip()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[^\w\s]', '', t)
    return t

def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()

def best_match(query_title: str, candidates: List[Dict[str, Any]], key="title") -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    scored = []
    for c in candidates:
        t = c.get(key) or ""
        scored.append((title_similarity(query_title, t), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None

def _map_illegal_to_fullwidth(ch: str) -> str:
    mapping = {
        '/': 'ï¼', '\\': 'ï¼¼', ':': 'ï¼š', '*': 'ï¼Š', '?': 'ï¼Ÿ', '"': 'ï¼‚', '<': 'ï¼œ', '>': 'ï¼ž', '|': 'ï½œ', '\0': ''
    }
    return mapping.get(ch, ch)

def sanitize_filename(name: str, max_len: int = 180, punct_mode: str = 'safe') -> str:
    """
    punct_mode:
      - 'safe' (default): map illegal chars to fullwidth twins, cross-platform safe
      - 'strip': remove illegal chars entirely
      - 'raw': keep punctuation as-is EXCEPT path separators (/, \\) and NUL
    """
    name = name.replace('{', '').replace('}', '')
    if punct_mode not in {'safe','strip','raw'}:
        punct_mode = 'safe'
    out = []
    for ch in name:
        if ch in ['/', '\\', '\x00']:
            # never allow path separators or NUL
            if punct_mode == 'raw':
                out.append('ï¼')  # replace with fullwidth slash
            elif punct_mode == 'strip':
                continue
            else:
                out.append(_map_illegal_to_fullwidth(ch))
        elif ch in [':', '*', '?', '"', '<', '>', '|']:
            if punct_mode == 'safe':
                out.append(_map_illegal_to_fullwidth(ch))
            elif punct_mode == 'strip':
                continue
            else:
                out.append(ch)  # raw
        else:
            out.append(ch)
    name = ''.join(out)
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name

def first_author_from_authors(authors_field: str) -> str:
    if not authors_field:
        return ""
    # Split by ' and ' (BibTeX author list)
    parts = [a.strip() for a in re.split(r'\s+and\s+', authors_field, flags=re.IGNORECASE) if a.strip()]
    if not parts:
        return ""
    first = parts[0]
    # Handle "Last, First" or "First Last"
    if "," in first:
        last = first.split(",")[0].strip()
        return last
    else:
        # take last token as surname
        tokens = first.split()
        return tokens[-1] if tokens else first

# -------------------- BibTeX parsing --------------------
# Minimal parser robust enough for typical .bib files. If you have bibtexparser installed, feel free to replace.

ENTRY_RE = re.compile(r'@(?P<type>\w+)\s*\{\s*(?P<citekey>[^,]+)\s*,', re.MULTILINE)
FIELD_RE = re.compile(r'(?P<field>\w+)\s*=\s*(?P<value>\{(?:[^{}]|\{[^{}]*\})*\}|"(?:[^"\\]|\\.)*"|[^,\n]+)\s*,?', re.MULTILINE)

def parse_bibtex(content: str) -> List[Dict[str, Any]]:
    entries = []
    pos = 0
    while True:
        m = ENTRY_RE.search(content, pos)
        if not m:
            break
        start = m.end()
        # find matching closing brace of entry
        brace_level = 1
        i = start
        while i < len(content) and brace_level > 0:
            if content[i] == "{":
                brace_level += 1
            elif content[i] == "}":
                brace_level -= 1
            i += 1
        entry_block = content[m.start():i]
        pos = i

        entry_type = m.group("type").strip()
        citekey = m.group("citekey").strip()

        fields = {}
        # search fields inside the block (after first comma)
        for fm in FIELD_RE.finditer(entry_block):
            field = fm.group("field").lower()
            value = fm.group("value").strip()
            # strip wrapping braces or quotes
            if value.startswith("{") and value.endswith("}"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # unescape latex-ish braces in titles
            value = value.replace("\n", " ").strip()
            fields[field] = value
        entry = {"type": entry_type, "citekey": citekey}
        entry.update(fields)
        entries.append(entry)
    return entries

# -------------------- Source lookups --------------------

def search_semanticscholar(title: str) -> Optional[Dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 5, "fields": "title,year,url,venue,openAccessPdf,externalIds,authors"}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        papers = data.get("data", [])
        match = best_match(title, papers, key="title")
        return match
    except Exception:
        return None

def search_arxiv(title: str) -> Optional[Dict[str, Any]]:
    q = f'all:"{title}"'
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": q, "start": 0, "max_results": 5}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=20)
        if r.status_code != 200:
            return None
        feed = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = feed.findall("atom:entry", ns)
        candidates = []
        for e in entries:
            title_el = e.find("atom:title", ns)
            if title_el is None:
                continue
            t = html.unescape(title_el.text or "").strip()
            pdf_link = None
            for link in e.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_link = link.attrib.get("href")
            candidates.append({"title": t, "pdf": pdf_link})
        match = best_match(title, candidates, key="title")
        return match
    except Exception:
        return None

def search_openalex(title: str) -> Optional[Dict[str, Any]]:
    url = "https://api.openalex.org/works"
    params = {"search": title, "per_page": 5}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get("results", [])
        candidates = []
        for w in results:
            t = w.get("title") or ""
            pdf = None
            pl = w.get("primary_location") or {}
            pdf = pl.get("pdf_url") or (w.get("open_access") or {}).get("oa_url")
            candidates.append({"title": t, "pdf": pdf, "doi": w.get("doi")})
        match = best_match(title, candidates, key="title")
        return match
    except Exception:
        return None

def unpaywall_pdf_from_doi(doi: str, email: Optional[str]) -> Optional[str]:
    if not doi or not email:
        return None
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}"
    try:
        r = requests.get(url, params={"email": email}, headers=UA, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        loc = data.get("best_oa_location") or {}
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
        for l in data.get("oa_locations", []):
            if l.get("url_for_pdf"):
                return l["url_for_pdf"]
        return None
    except Exception:
        return None

def resolve_pdf_for_entry(title: str, doi: Optional[str], arxiv_id: Optional[str], unpaywall_email: Optional[str]) -> Dict[str, Any]:
    """Return dict: {'source','pdf_url','meta':{...}}"""
    # Direct arXiv by ID if present
    if arxiv_id:
        return {"source": "arXiv(id)", "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf", "meta": {}}

    # Semantic Scholar first
    ss = search_semanticscholar(title)
    if ss:
        oapdf = (ss.get("openAccessPdf") or {}).get("url")
        if oapdf:
            return {"source": "SemanticScholar", "pdf_url": oapdf, "meta": {"venue": ss.get("venue"), "year": ss.get("year")}}
        ex = ss.get("externalIds") or {}
        if ex.get("ArXiv"):
            return {"source": "SemanticScholar(arXiv)", "pdf_url": f"https://arxiv.org/pdf/{ex['ArXiv']}.pdf", "meta": {"venue": ss.get("venue"), "year": ss.get("year")}}
        if ex.get("DOI"):
            doi = doi or ex.get("DOI")

    # arXiv by title
    ax = search_arxiv(title)
    if ax and ax.get("pdf"):
        return {"source": "arXiv", "pdf_url": ax["pdf"], "meta": {}}

    # OpenAlex
    oa = search_openalex(title)
    if oa and oa.get("pdf"):
        return {"source": "OpenAlex", "pdf_url": oa["pdf"], "meta": {"doi": oa.get("doi")}}
    if doi is None and oa and oa.get("doi"):
        doi = oa["doi"]

    # Unpaywall via DOI (optional)
    if doi:
        pdf = unpaywall_pdf_from_doi(doi, unpaywall_email)
        if pdf:
            return {"source": "Unpaywall", "pdf_url": pdf, "meta": {"doi": doi}}

    return {"source": None, "pdf_url": None, "meta": {}}

def http_get(url: str, timeout=30) -> Optional[bytes]:
    try:
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and ("application/pdf" in r.headers.get("Content-Type", "").lower() or url.lower().endswith(".pdf")):
            return r.content
        # Some servers may send octet-stream; accept it
        if r.status_code == 200 and "application/octet-stream" in r.headers.get("Content-Type", "").lower():
            return r.content
        return None
    except Exception:
        return None

# -------------------- Main workflow --------------------

def make_filename(entry: Dict[str, Any], template: str, punct_mode: str) -> str:
    title = entry.get("title") or entry.get("booktitle") or entry.get("journal") or "untitled"
    authors = entry.get("author", "")
    year = entry.get("year", "")
    first_author = first_author_from_authors(authors)
    citekey = entry.get("citekey", "nocitekey")

    # flatten LaTeX braces and TeX accents minimally
    title_clean = re.sub(r'[\{\}]', '', title)
    title_clean = re.sub(r'\\[a-zA-Z]+\s*', '', title_clean)  # drop simple macros

    name = template.format(
        citekey=citekey,
        title=title_clean,
        year=year,
        first_author=first_author
    )
    name = sanitize_filename(name, punct_mode=punct_mode)
    return name + ".pdf"

def main():
    parser = argparse.ArgumentParser(description="Download PDFs for BibTeX entries and name files from a template.")
    parser.add_argument("--bib", required=True, help="Path to .bib file")
    parser.add_argument("--outdir", default="./pdfs", help="Output directory for PDFs")
    parser.add_argument("--template", default="{citekey}", help="Filename template; fields: {citekey},{title},{year},{first_author}")
    parser.add_argument("--punct", choices=["safe","strip","raw"], default="safe", help="How to handle punctuation/illegal chars in filenames")
    parser.add_argument("--sleep", type=float, default=0.6, help="Sleep seconds between network calls (politeness)")
    parser.add_argument("--dry-run", action="store_true", help="Do not download; just resolve and print plan")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files if present")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    email = os.environ.get("UNPAYWALL_EMAIL")

    with open(args.bib, "r", encoding="utf-8") as f:
        content = f.read()

    entries = parse_bibtex(content)
    if not entries:
        print("No BibTeX entries found.")
        return

    manifest_path = os.path.join(args.outdir, "paper_pdf_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["citekey","title","year","first_author","doi","arxiv_id","source","pdf_url","filename","status","message"])
        writer.writeheader()

        for idx, entry in enumerate(entries, 1):
            title = entry.get("title") or ""
            doi = entry.get("doi") or entry.get("DOI")
            eprint = entry.get("eprint") or entry.get("arxivid") or ""
            arxiv_id = None
            # Try to normalize arXiv id
            if eprint:
                m = re.match(r'(?:arXiv:)?(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})', eprint, re.IGNORECASE)
                if m:
                    arxiv_id = m.group(1)

            filename = make_filename(entry, args.template, args.punct)
            outpath = os.path.join(args.outdir, filename)

            print(f"[{idx}/{len(entries)}] Resolving: {entry.get('citekey','?')} :: {title[:80]}")
            resolved = resolve_pdf_for_entry(title=title, doi=doi, arxiv_id=arxiv_id, unpaywall_email=email)
            pdf_url = resolved.get("pdf_url")
            source = resolved.get("source")

            status = "skipped"
            message = ""
            if os.path.exists(outpath) and not args.overwrite:
                status = "exists"
                message = "File exists; skipping"
            elif not pdf_url:
                status = "not_found"
                message = "No PDF URL found from sources"
            else:
                if args.dry_run:
                    status = "dry_run"
                    message = f"Would download from {source}"
                else:
                    blob = http_get(pdf_url)
                    if blob:
                        with open(outpath, "wb") as outf:
                            outf.write(blob)
                        status = "downloaded"
                        message = f"Downloaded from {source}"
                    else:
                        status = "download_failed"
                        message = "HTTP failed or not a PDF"

            writer.writerow({
                "citekey": entry.get("citekey",""),
                "title": title,
                "year": entry.get("year",""),
                "first_author": first_author_from_authors(entry.get("author","")),
                "doi": doi or "",
                "arxiv_id": arxiv_id or "",
                "source": source or "",
                "pdf_url": pdf_url or "",
                "filename": filename,
                "status": status,
                "message": message
            })
            time.sleep(args.sleep)

    print(f"\nDone. Manifest saved to: {manifest_path}")
    print(f"PDFs (if downloaded) are in: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
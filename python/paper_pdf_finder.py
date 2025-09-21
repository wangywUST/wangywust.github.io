
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_pdf_finder.py
Find open-access PDFs for academic papers by title using public APIs:
- Semantic Scholar (primary): openAccessPdf, arXiv/DOI fallbacks
- arXiv API (for preprints)
- OpenAlex (extra coverage)
Optional: Unpaywall for DOI OA resolution (set env UNPAYWALL_EMAIL).
"""
import os
import sys
import time
import json
import csv
import argparse
import html
import re
from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher
import urllib.parse
import xml.etree.ElementTree as ET

try:
    import requests
except Exception as e:
    print("This script requires the 'requests' package. Install with: pip install requests")
    sys.exit(1)

UA = {"User-Agent": "pdf-finder/1.0 (https://github.com/)"}  # adjust as you like

def norm_title(t: str) -> str:
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
    top = scored[0][1] if scored else None
    return top

# -------- Semantic Scholar --------
def search_semanticscholar(title: str) -> Optional[Dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 5,
        "fields": "title,year,url,venue,openAccessPdf,externalIds,authors"
    }
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

# -------- arXiv --------
def search_arxiv(title: str) -> Optional[Dict[str, Any]]:
    q = f'all:"{title}"'
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": q, "start": 0, "max_results": 5}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=20)
        if r.status_code != 200:
            return None
        feed = ET.fromstring(r.text)
        # Namespaces
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

# -------- OpenAlex --------
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
            # primary_location or open_access.open_access_url
            pl = w.get("primary_location") or {}
            pdf = (pl.get("pdf_url")
                   or (pl.get("source") or {}).get("host_organization_page")
                   or (w.get("open_access") or {}).get("oa_url"))
            candidates.append({"title": t, "pdf": pdf, "doi": w.get("doi")})
        match = best_match(title, candidates, key="title")
        return match
    except Exception:
        return None

# -------- Unpaywall (optional) --------
def unpaywall_pdf_from_doi(doi: str, email: Optional[str]) -> Optional[str]:
    if not doi or not email:
        return None
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}"
    try:
        r = requests.get(url, params={"email": email}, headers=UA, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        # Try best_oa_location, then oa_locations
        loc = data.get("best_oa_location") or {}
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
        for l in data.get("oa_locations", []):
            if l.get("url_for_pdf"):
                return l["url_for_pdf"]
        return None
    except Exception:
        return None

def resolve_pdf_for_title(title: str, unpaywall_email: Optional[str] = None) -> Dict[str, Any]:
    """Return dict with keys: title, source, pdf_url, extra"""
    # 1) Semantic Scholar
    ss = search_semanticscholar(title)
    if ss:
        # Exact OA PDF?
        oapdf = (ss.get("openAccessPdf") or {}).get("url")
        if oapdf:
            return {"title": ss.get("title", title), "source": "SemanticScholar", "pdf_url": oapdf, "extra": {"venue": ss.get("venue"), "year": ss.get("year")}}
        # arXiv via externalIds
        ex = ss.get("externalIds") or {}
        if ex.get("ArXiv"):
            return {"title": ss.get("title", title), "source": "SemanticScholar(arXiv)", "pdf_url": f"https://arxiv.org/pdf/{ex['ArXiv']}.pdf", "extra": {"venue": ss.get("venue"), "year": ss.get("year")}}
        # DOI via Unpaywall (optional)
        if ex.get("DOI"):
            pdf = unpaywall_pdf_from_doi(ex["DOI"], unpaywall_email)
            if pdf:
                return {"title": ss.get("title", title), "source": "SemanticScholar(Unpaywall)", "pdf_url": pdf, "extra": {"venue": ss.get("venue"), "year": ss.get("year")}}
    # 2) arXiv API
    ax = search_arxiv(title)
    if ax and ax.get("pdf"):
        return {"title": ax.get("title", title), "source": "arXiv", "pdf_url": ax["pdf"], "extra": {}}
    # 3) OpenAlex
    oa = search_openalex(title)
    if oa and oa.get("pdf"):
        return {"title": oa.get("title", title), "source": "OpenAlex", "pdf_url": oa["pdf"], "extra": {"doi": oa.get("doi")}}
    # 4) Fallback: try Unpaywall with DOI from OpenAlex or elsewhere
    if oa and oa.get("doi"):
        pdf = unpaywall_pdf_from_doi(oa["doi"], unpaywall_email)
        if pdf:
            return {"title": oa.get("title", title), "source": "Unpaywall", "pdf_url": pdf, "extra": {"doi": oa.get("doi")}}
    return {"title": title, "source": None, "pdf_url": None, "extra": {}}

def main():
    parser = argparse.ArgumentParser(description="Find open-access PDF URLs by paper title (Semantic Scholar, arXiv, OpenAlex; optional Unpaywall).")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--title", type=str, help="Single paper title")
    g.add_argument("--file", type=str, help="Path to a text file with one title per line")
    parser.add_argument("--out", type=str, help="Optional: write CSV results to this file")
    parser.add_argument("--sleep", type=float, default=0.6, help="Sleep seconds between queries to be polite")
    args = parser.parse_args()

    email = os.environ.get("UNPAYWALL_EMAIL")

    titles = []
    if args.title:
        titles = [args.title.strip()]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            titles = [line.strip() for line in f if line.strip()]

    results = []
    for t in titles:
        res = resolve_pdf_for_title(t, unpaywall_email=email)
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))
        time.sleep(args.sleep)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["title", "source", "pdf_url", "extra"])
            w.writeheader()
            for r in results:
                r2 = r.copy()
                r2["extra"] = json.dumps(r2.get("extra", {}), ensure_ascii=False)
                w.writerow(r2)

if __name__ == "__main__":
    main()

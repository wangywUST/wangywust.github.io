# -*- coding: utf-8 -*-
"""Try to download the two missing PDFs: gao2026orbitpsibench, nan2026scti.
Priority: arXiv API -> Semantic Scholar openAccessPdf.
Writes results to stdout as JSON lines.
"""
import json, ssl, time, urllib.parse, urllib.request, os, sys

try:
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ctx = ssl._create_unverified_context()

PDF_DIR = r"E:\wangywust.github.io\pdfs"
UA = {"User-Agent": "Mozilla/5.0 (research-pdf-fetcher; mailto:wangyw@example.com)"}

def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read(), dict(r.headers)

def try_arxiv(title):
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"https://export.arxiv.org/api/query?search_query={q}&max_results=3"
    try:
        body, _ = http_get(url, timeout=30)
        txt = body.decode("utf-8", "ignore")
    except Exception as e:
        return {"ok": False, "err": f"arxiv api: {e}"}
    # crude parse: find <entry>...<id>http://arxiv.org/abs/XXXX.XXXXX
    import re
    entries = re.findall(r"<entry>(.*?)</entry>", txt, re.S)
    out = []
    for e in entries:
        m_id = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
        m_ti = re.search(r"<title>(.*?)</title>", e, re.S)
        if m_id:
            aid = m_id.group(1).strip()
            atitle = re.sub(r"\s+", " ", m_ti.group(1)).strip() if m_ti else ""
            out.append({"arxiv_id": aid, "title": atitle})
    return {"ok": True, "results": out}

def try_s2(title):
    q = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&fields=title,openAccessPdf,externalIds,venue,year&limit=5"
    for attempt in range(2):
        try:
            body, _ = http_get(url, timeout=30)
            data = json.loads(body.decode("utf-8", "ignore"))
            return {"ok": True, "data": data}
        except Exception as e:
            if attempt == 0:
                time.sleep(5)
            else:
                return {"ok": False, "err": f"s2 api: {e}"}

def download_pdf(url, dest):
    tmp = dest + ".tmp"
    try:
        body, headers = http_get(url, timeout=120)
    except Exception as e:
        return False, f"download: {e}"
    if not body[:5] == b"%PDF-":
        return False, f"not a PDF (header={body[:20]!r}, {len(body)} bytes)"
    if len(body) < 20000:
        return False, f"too small ({len(body)} bytes)"
    with open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, dest)
    return True, f"ok {len(body)} bytes"

papers = [
    {
        "citekey": "gao2026orbitpsibench",
        "title": "OrbitPSIBench: Benchmarking Vision-Free LLMs' Anatomical Reasoning in Orbital Bone",
    },
    {
        "citekey": "nan2026scti",
        "title": "SCTI: Self-Calibrated Trident Identification of Black-Box LLM Watermarks",
    },
]

for p in papers:
    print(f"=== {p['citekey']}: {p['title']}")
    ar = try_arxiv(p["title"])
    print("arXiv:", json.dumps(ar, ensure_ascii=False))
    time.sleep(3)
    s2 = try_s2(p["title"])
    print("S2:", json.dumps(s2, ensure_ascii=False)[:2000])
    time.sleep(3)
    # try S2 openAccessPdf if present
    if s2.get("ok"):
        for paper in s2["data"].get("data", []):
            oap = paper.get("openAccessPdf")
            if oap and oap.get("url"):
                dest = os.path.join(PDF_DIR, p["citekey"] + ".pdf")
                print(f"  -> trying S2 openAccessPdf: {oap['url']}")
                ok, msg = download_pdf(oap["url"], dest)
                print(f"     {ok}: {msg}")
                if ok:
                    break
    # try arxiv pdf if any returned
    if ar.get("ok"):
        for r in ar.get("results", []):
            dest = os.path.join(PDF_DIR, p["citekey"] + ".pdf")
            if os.path.exists(dest):
                break
            pdf_url = f"https://arxiv.org/pdf/{r['arxiv_id']}.pdf"
            print(f"  -> trying arXiv: {pdf_url} (title={r['title']})")
            ok, msg = download_pdf(pdf_url, dest)
            print(f"     {ok}: {msg}")
            if ok:
                break

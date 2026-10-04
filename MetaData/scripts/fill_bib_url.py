import re
import time
import html
import difflib
import unicodedata
import xml.etree.ElementTree as ET

import requests
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from common import DATA

MAILTO = "your_email@example.com"   # Crossref polite pool，填真实邮箱更稳定
S2_API_KEY = None                    # 可选：Semantic Scholar API key，限速宽松很多

SESSION = requests.Session()
SESSION.headers["User-Agent"] = f"bib-url-filler (mailto:{MAILTO})"


# ---------- 标题归一化与匹配 ----------
def norm(t):
    t = html.unescape(t or "")
    t = re.sub(r"<[^>]+>", " ", t)            # Crossref 的 <i>、<sub> 等标签
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)      # LaTeX 命令
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return " ".join(t.split())


def same_title(a, b, th=0.92):
    a, b = norm(a), norm(b)
    return bool(a) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= th)


def get_json(url, params=None, headers=None, retries=3):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                last = "429 rate limited"
                time.sleep(3 * 2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(2 ** i)
    print(f"  [warn] {url}: {last}")
    return None


# ---------- 0. 本地字段 ----------
ARXIV_RE = re.compile(r"(?:arxiv[:\s/]*|abs/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def local_url(e):
    if e.get("doi"):
        return "https://doi.org/" + e["doi"].strip()
    eprint = e.get("eprint", "").strip()
    if eprint and (e.get("archiveprefix", "").lower() == "arxiv"
                   or re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", eprint)):
        return "https://arxiv.org/abs/" + re.sub(r"v\d+$", "", eprint)
    for f in ("journal", "volume", "note", "howpublished", "booktitle"):
        m = ARXIV_RE.search(e.get(f, ""))
        if m:
            return "https://arxiv.org/abs/" + m.group(1)
    return None


# ---------- 1. DBLP：CS 会议覆盖最好，直接给官方链接 ----------
def from_dblp(title):
    d = get_json("https://dblp.org/search/publ/api",
                 {"q": norm(title), "format": "json", "h": 10})
    hits = ((d or {}).get("result", {}).get("hits", {}).get("hit", []))
    cands = []
    for h in hits:
        info = h.get("info", {})
        if not same_title(title, info.get("title", "")):
            continue
        ee = info.get("ee")
        cands += ee if isinstance(ee, list) else ([ee] if ee else [])
    cands.sort(key=lambda u: "arxiv.org" in u)   # 正式发表版本优先
    return cands[0] if cands else None


# ---------- 2. Semantic Scholar：标题匹配端点 ----------
def from_s2(title):
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else None
    d = get_json("https://api.semanticscholar.org/graph/v1/paper/search/match",
                 {"query": title, "fields": "title,externalIds,url"}, headers=headers)
    for p in (d or {}).get("data", []):
        if not same_title(title, p.get("title", "")):
            continue
        ids = p.get("externalIds") or {}
        if ids.get("ACL"):
            return f"https://aclanthology.org/{ids['ACL']}"
        if ids.get("DOI") and not ids["DOI"].lower().startswith("10.48550"):
            return f"https://doi.org/{ids['DOI']}"
        if ids.get("ArXiv"):
            return f"https://arxiv.org/abs/{ids['ArXiv']}"
        return p.get("url")
    return None


# ---------- 3. Crossref：期刊 / IEEE / ACM / Springer ----------
def from_crossref(title):
    d = get_json("https://api.crossref.org/works",
                 {"query.bibliographic": title, "rows": 5,
                  "select": "DOI,title", "mailto": MAILTO})
    for it in (d or {}).get("message", {}).get("items", []):
        if same_title(title, (it.get("title") or [""])[0]):
            return "https://doi.org/" + it["DOI"]
    return None


# ---------- 4. arXiv API：最后兜底 ----------
def from_arxiv(title):
    try:
        r = SESSION.get("https://export.arxiv.org/api/query",
                        params={"search_query": f'ti:"{norm(title)}"', "max_results": 5},
                        timeout=20)
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"  [warn] arXiv: {e}")
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for ent in root.findall("a:entry", ns):
        if same_title(title, ent.findtext("a:title", "", ns)):
            aid = ent.findtext("a:id", "", ns)
            return re.sub(r"v\d+$", "", aid).replace("http://", "https://")
    return None


# (名称, 函数, 调用后等待秒数) —— 按礼貌限速设置
SOURCES = [
    ("dblp", from_dblp, 1.0),
    ("s2", from_s2, 1.1),
    ("crossref", from_crossref, 0.2),
    ("arxiv", from_arxiv, 3.0),
]


def fill_bib_urls(bib_path, out_path=None):
    with open(bib_path, encoding="utf-8") as f:
        db = bibtexparser.load(f)

    failed = []
    for e in db.entries:
        if e.get("url"):
            continue
        title = e.get("title", "")
        url, src = local_url(e), "local"
        if not url and title:
            for name, fn, wait in SOURCES:
                url = fn(title)
                time.sleep(wait)
                if url:
                    src = name
                    break
        if url:
            e["url"] = url
            print(f"[{src:8}] {e['ID']}: {url}")
        else:
            failed.append(e["ID"])
            print(f"[MISS    ] {e['ID']}: {title}")

    writer = BibTexWriter()
    writer.indent = "  "
    with open(out_path or bib_path, "w", encoding="utf-8") as f:
        f.write(writer.write(db))

    if failed:
        print(f"\n{len(failed)} 条未找到，需手动检查: {', '.join(failed)}")


if __name__ == "__main__":
    # 先输出到新文件，确认无误再覆盖原文件
    fill_bib_urls(DATA / "citations.bib", DATA / "citations.filled.bib")
# -*- coding: utf-8 -*-
"""Step 2: extract first-page text from each available PDF, try to isolate
the Abstract paragraph. Save everything to tmp/pdf_texts.json for LLM review."""
import json, os, re, warnings
warnings.filterwarnings("ignore")

from pypdf import PdfReader

ROOT = r"E:\wangywust.github.io"
PDFDIR = os.path.join(ROOT, "pdfs")
ENTRIES = os.path.join(ROOT, "tmp", "entries.json")
OUT = os.path.join(ROOT, "tmp", "pdf_texts.json")

with open(ENTRIES, encoding="utf-8") as f:
    entries = json.load(f)

ABSTRACT_RE = re.compile(
    r"(?is)\babstract\b[\s:\-\u2014]*\s*(.+?)"
    r"(?=\n\s*(?:1[\.\s]+introduction|i\.[\s]+introduction|introduction\b"
    r"|index terms|keywords|ccs\s*concepts|categories and subject))"
)

out = {}
for e in entries:
    if e["pdf_status"] != "ok":
        continue
    key = e["key"]
    path = os.path.join(PDFDIR, key + ".pdf")
    rec = {"title": e["title"], "venue": e.get("venue", "")}
    try:
        reader = PdfReader(path)
        text = reader.pages[0].extract_text() or ""
        if len(text.strip()) < 500 and len(reader.pages) > 1:
            text += "\n" + (reader.pages[1].extract_text() or "")
        rec["n_pages"] = len(reader.pages)
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"
        out[key] = rec
        continue

    txt = text.replace("\r", "")
    rec["raw_len"] = len(txt)
    m = ABSTRACT_RE.search(txt)
    if m:
        ab = m.group(1).strip()
        ab = re.sub(r"\s+", " ", ab)
        rec["abstract_candidate"] = ab[:2500]
    else:
        # fallback: everything after the first 'Abstract' heading (on its own)
        m2 = re.search(r"(?is)^\s*abstract\s*:?\s*(.+)", txt, flags=re.M)
        if m2:
            ab = m2.group(1)[:1800]
            ab = re.sub(r"\s+", " ", ab)
            rec["abstract_candidate"] = ab
    # also keep a first-2000 snippet for fallback judgment
    rec["snippet"] = re.sub(r"\n{3,}", "\n\n", txt[:2000])
    out[key] = rec

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# stats
got = [k for k, v in out.items() if v.get("abstract_candidate")]
err = [k for k, v in out.items() if v.get("error")]
thin = [k for k, v in out.items() if not v.get("abstract_candidate") and not v.get("error")]
print(f"processed PDFs: {len(out)}")
print(f"abstract regex hit: {len(got)}")
print(f"extract error: {len(err)} -> {err}")
print(f"no regex hit (need snippet review): {len(thin)}")
for k in thin:
    print("   ", k, "raw_len=", out[k].get("raw_len"))

# -*- coding: utf-8 -*-
"""Step 1: parse citations.bib, list entries, detect abstract presence,
and check PDF availability. Output JSON manifest to tmp/entries.json."""
import json, os, re
from collections import Counter
import bibtexparser

ROOT = r"E:\wangywust.github.io"
BIB = os.path.join(ROOT, "MetaData", "data", "citations.bib")
PDFDIR = os.path.join(ROOT, "pdfs")
OUT = os.path.join(ROOT, "tmp", "entries.json")

with open(BIB, "r", encoding="utf-8") as f:
    raw = f.read()

lib = bibtexparser.parse_string(raw)
entries = lib.entries
print(f"Total entries: {len(entries)}")

# Build citekey -> field dict
def fields_of(e):
    d = {}
    for fld in e.fields:
        d[fld.key.lower()] = fld.value
    return d

# Locate entry blocks in raw text by brace matching.
blocks = {}
i, n = 0, len(raw)
while True:
    m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,").search(raw, i)
    if not m:
        break
    etype, key = m.group(1), m.group(2)
    brace_start = raw.index("{", m.start())
    depth = 0; j = brace_start; in_str = False
    while j < n:
        c = raw[j]
        if c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    blocks[key] = (brace_start, j + 1, etype)
    i = j + 1

result = []
for e in entries:
    key = e.key
    flds = fields_of(e)
    title = flds.get("title", "").strip()
    has_abs = bool(flds.get("abstract", "").strip())
    start, end, etype = blocks.get(key, (None, None, None))
    info = {
        "key": key, "title": title, "type": etype,
        "has_abstract": has_abs,
        "bib_start": start, "bib_end": end,
        "venue": flds.get("booktitle") or flds.get("journal") or "",
    }
    if start is not None:
        block = raw[start:end]
        tm = re.search(r"^([ \t]*)title(\s*)=", block, flags=re.M)
        if tm:
            line_start = start + tm.start()
            line_end = raw.find("\n", line_start)
            line = raw[line_start:line_end]
            info["title_line_start"] = line_start
            info["title_line_end"] = line_end
            info["title_indent"] = tm.group(1)
            info["title_style"] = "space" if tm.group(2) else "nospace"
        else:
            info["title_line_start"] = None
    result.append(info)

for r in result:
    p = os.path.join(PDFDIR, r["key"] + ".pdf")
    if not os.path.exists(p):
        r["pdf_status"] = "missing"
    else:
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            head = f.read(5)
        if head[:4] != b"%PDF":
            r["pdf_status"] = "bad_header"
        elif size < 20 * 1024:
            r["pdf_status"] = "too_small"
        else:
            r["pdf_status"] = "ok"
        r["pdf_size"] = size

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=1)

need = [r for r in result if not r["has_abstract"]]
print(f"Already have abstract: {len(result)-len(need)}")
print(f"Need abstract: {len(need)}")
print("PDF status among need-abstract entries:", Counter(r["pdf_status"] for r in need))
print()
for r in need:
    print(f"  [{r['pdf_status']:10s}] {r['key']:32s} | {r['title'][:65]}")

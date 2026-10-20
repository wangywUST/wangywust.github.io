# -*- coding: utf-8 -*-
"""Step 4: merge per-batch abstract JSONs and surgically insert abstract fields
into citations.bib right after each entry's title line."""
import json, os, re, glob

ROOT = r"E:\wangywust.github.io"
BIB = os.path.join(ROOT, "MetaData", "data", "citations.bib")
ENTRIES = os.path.join(ROOT, "tmp", "entries.json")

# 1. merge abstracts
abstracts = {}
for f in sorted(glob.glob(os.path.join(ROOT, "tmp", "abs", "batch*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    abstracts.update(d)
print(f"merged abstracts: {len(abstracts)}")

entries = json.load(open(ENTRIES, encoding="utf-8"))
info = {e["key"]: e for e in entries}

# 2. sanity: every abstract key exists in bib, has title line info, has no abstract yet
problems = []
for k in abstracts:
    e = info.get(k)
    if e is None:
        problems.append((k, "not in bib")); continue
    if e.get("title_line_start") is None:
        problems.append((k, "no title line located")); continue
    if e.get("has_abstract"):
        problems.append((k, "already has abstract - skip")); continue
if problems:
    for p in problems: print("PROBLEM:", p)
    raise SystemExit("fix problems before inserting")

# 3. escape special chars inside abstract text
def esc(s):
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s

# 4. read bib raw
with open(BIB, "r", encoding="utf-8") as f:
    raw = f.read()

# build insertions: (offset_after_title_line, text_to_insert)
inserts = []
for k, ab in abstracts.items():
    e = info[k]
    ab_esc = esc(ab.strip())
    indent = e.get("title_indent") or "  "
    eq = " = " if e.get("title_style") == "space" else "="
    line = f"{indent}abstract{eq}{{{ab_esc}}},\n"
    inserts.append((e["title_line_end"], line))  # insert right after title line's newline position

# sort by offset descending so earlier offsets stay valid
inserts.sort(key=lambda x: -x[0])

# sanity check: inserted braces balanced
for off, line in inserts:
    assert line.count("{") == line.count("}"), line[:80]

new = raw
for off, line in inserts:
    new = new[:off+1] + line + new[off+1:]  # off is index of '\n' after title; keep it, insert after

# backup original, then write
import shutil
shutil.copy(BIB, BIB + ".bak")
with open(BIB, "w", encoding="utf-8") as f:
    f.write(new)

print(f"inserted {len(inserts)} abstract fields")
print("backup at citations.bib.bak")

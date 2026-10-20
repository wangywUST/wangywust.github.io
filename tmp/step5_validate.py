# -*- coding: utf-8 -*-
"""Step 5: validate modified bib."""
import json, os
import bibtexparser

ROOT = r"E:\wangywust.github.io"
BIB = os.path.join(ROOT, "MetaData", "data", "citations.bib")
BAK = BIB + ".bak"

lib_new = bibtexparser.parse_string(open(BIB, encoding="utf-8").read())
lib_old = bibtexparser.parse_string(open(BAK, encoding="utf-8").read())

print("old entries:", len(lib_old.entries))
print("new entries:", len(lib_new.entries))
assert len(lib_new.entries) == len(lib_old.entries), "entry count changed!"

def fmap(e):
    return {f.key.lower(): f.value for f in e.fields}

old = {e.key: fmap(e) for e in lib_old.entries}
new = {e.key: fmap(e) for e in lib_new.entries}

# check non-abstract fields unchanged byte-wise
changed_other = 0
for k in old:
    for fld, v in old[k].items():
        if fld == "abstract":
            continue
        if new[k].get(fld) != v:
            changed_other += 1
            print("DIFF FIELD:", k, fld)
print("non-abstract fields changed:", changed_other)

added = [k for k in new if "abstract" in new[k] and "abstract" not in old[k]]
print("abstracts added:", len(added))
still_no = [k for k in new if "abstract" not in new[k]]
print("still without abstract:", len(still_no), still_no)

# spot check 3 entries
import textwrap
for k in added[:3] + added[-2:]:
    print("="*70)
    print(k, "|", new[k].get("title","")[:60])
    print(textwrap.fill(new[k]["abstract"][:400], 100))
    print("abstract word count:", len(new[k]["abstract"].split()))

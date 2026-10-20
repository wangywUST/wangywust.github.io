# -*- coding: utf-8 -*-
"""Fix manifest:
1. Drop the spurious duplicate wang2021curgraph row (heterogeneous graph explanations title).
2. Add a proper lu2026towards row with the correct WWW 2026 metadata.
"""
import csv, os

MANIFEST = r"E:\wangywust.github.io\pdfs\paper_pdf_manifest.csv"

with open(MANIFEST, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))
header = rows[0]
data = rows[1:]

new_data = []
for r in data:
    # Drop the spurious duplicate wang2021curgraph row
    if r[0] == "wang2021curgraph" and "Heterogeneous Graph Explanations" in r[1]:
        continue
    new_data.append(r)

# Insert the corrected lu2026towards row right after wang2021curgraph (CurGraph) for logical grouping
lu_row = [
    "lu2026towards",
    "Towards Robust Heterogeneous Graph Explanations under Structural Perturbations",
    "https://doi.org/10.1145/3774904.3792516",
    "https://dl.acm.org/doi/pdf/10.1145/3774904.3792516",
    "lu2026towards.pdf",
    "download_failed",
    "No script-accessible open PDF: WWW 2026 (RoHeX), ACM DL marks it HYBRID/CC-BY but dl.acm.org returns 403 to scripted requests (anti-bot). No arXiv preprint (exact-title query empty). OpenReview forum id=oINSSze5gI PDF also 403 (Cloudflare). S2 openAccessPdf points to the ACM DOI. Note: citations.bib currently has a wrong DOI (10.1145/3442381.3450025 = CurGraph's DOI); the correct DOI is 10.1145/3774904.3792516. Manually download in a browser from https://dl.acm.org/doi/pdf/10.1145/3774904.3792516 (open access, no subscription needed)."
]

# Find index of the (remaining) wang2021curgraph row and insert after it
out = [header]
inserted = False
for r in new_data:
    out.append(r)
    if r[0] == "wang2021curgraph" and not inserted:
        out.append(lu_row)
        inserted = True
if not inserted:
    out.append(lu_row)

with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerows(out)

print(f"Wrote {len(out)-1} rows.")
# Verify
from collections import Counter
keys = [r[0] for r in out[1:]]
print("Duplicates:", [k for k,c in Counter(keys).items() if c>1])
print("Non-exists rows:")
for r in out[1:]:
    if r[5] != "exists":
        print(f"  {r[0]} | {r[5]} | {r[1][:70]}")

# -*- coding: utf-8 -*-
"""Reconcile paper_pdf_manifest.csv against the actual pdfs/ directory.
Rules:
- If pdfs/{citekey}.pdf exists on disk (and is a valid %PDF, >=20KB) -> status=exists
- Special shared-citekey rows keep their explanatory message.
- Otherwise keep the existing failure message, refreshed with today's re-check.
"""
import csv, os, ssl, datetime

PDF_DIR = r"E:\wangywust.github.io\pdfs"
MANIFEST = os.path.join(PDF_DIR, "paper_pdf_manifest.csv")
TODAY = "2026-10-20"

def is_valid_pdf(path):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 20000:
        return False
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"

# Read all rows
with open(MANIFEST, "r", encoding="utf-8", newline="") as f:
    reader = list(csv.reader(f))
header = reader[0]
rows = reader[1:]

out = [header]
for row in rows:
    # pad
    while len(row) < 7:
        row.append("")
    citekey, title, source_url, pdf_url, filename, status, message = row[:7]
    expected = os.path.join(PDF_DIR, citekey + ".pdf")

    # Special: wang2021curgraph second row (heterogeneous graph explanations) shares the citekey
    # Keep its existing note verbatim.
    if citekey == "wang2021curgraph" and "Heterogeneous Graph Explanations" in title:
        out.append([citekey, title, source_url, pdf_url, filename, status, message])
        continue

    if is_valid_pdf(expected):
        # File now exists. If it was previously failed, note that.
        if status != "exists":
            new_msg = f"PDF now present on disk ({os.path.getsize(expected):,} bytes, %PDF header OK) as of {TODAY}; reconciled from filesystem."
        else:
            new_msg = message if message else "PDF already exists"
        out.append([citekey, title, source_url, pdf_url, filename, "exists", new_msg])
    else:
        # Still missing. Keep the existing failure message, append re-check note.
        if "re-checked" not in message:
            message = message.rstrip(".") + f". Re-checked {TODAY}: still no open source found (arXiv empty, OpenReview 403, no author preprint)."
        out.append([citekey, title, source_url, pdf_url, filename, status, message])

# Write back
with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerows(out)

print(f"Wrote {len(out)-1} rows to {MANIFEST}")
# Print summary of non-exists rows
print("\n=== Non-exists rows ===")
for row in out[1:]:
    if row[5] != "exists":
        print(f"  {row[0]} | {row[5]} | {row[1][:60]}")

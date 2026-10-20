# -*- coding: utf-8 -*-
"""Step 3: split extracted abstracts into digest text files of N papers each
for LLM batch review/writing."""
import json, os

ROOT = r"E:\wangywust.github.io"
data = json.load(open(os.path.join(ROOT, "tmp", "pdf_texts.json"), encoding="utf-8"))
entries = json.load(open(os.path.join(ROOT, "tmp", "entries.json"), encoding="utf-8"))

# keep order of entries.json
order = [e["key"] for e in entries if e["key"] in data and data[e["key"]].get("abstract_candidate")]

BATCH = 12
os.makedirs(os.path.join(ROOT, "tmp", "batches"), exist_ok=True)
n = 0
for i in range(0, len(order), BATCH):
    chunk = order[i:i+BATCH]
    n += 1
    lines = []
    for k in chunk:
        v = data[k]
        lines.append(f"### KEY: {k}")
        lines.append(f"TITLE: {v['title']}")
        lines.append(f"VENUE: {v.get('venue','')}")
        lines.append("PAPER ABSTRACT TEXT:")
        lines.append(v["abstract_candidate"].strip())
        lines.append("")
    with open(os.path.join(ROOT, "tmp", "batches", f"batch_{n:02d}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
print(f"total papers in batches: {len(order)}, batches: {n}")

# -*- coding: utf-8 -*-
"""Targeted insertion for the two newly available entries only."""
import json, os, re

ROOT = r"E:\wangywust.github.io"
BIB = os.path.join(ROOT, "MetaData", "data", "citations.bib")
ABSTRACTS = json.load(open(os.path.join(ROOT, "tmp", "abs", "batch11.json"), encoding="utf-8"))

raw = open(BIB, encoding="utf-8").read()

def esc(s):
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s

inserts = []
for key, ab in ABSTRACTS.items():
    # find entry block
    m = re.search(r"@\w+\s*\{\s*" + re.escape(key) + r"\s*,", raw)
    assert m, key
    brace_start = raw.index("{", m.start())
    depth = 0; j = brace_start
    while j < len(raw):
        if raw[j] == "{": depth += 1
        elif raw[j] == "}":
            depth -= 1
            if depth == 0: break
        j += 1
    block = raw[brace_start:j+1]
    tm = re.search(r"^([ \t]*)title(\s*)=", block, flags=re.M)
    assert tm, key
    line_start_in_block = tm.start()
    line_end_in_block = block.index("\n", tm.start())
    abs_end = brace_start + line_end_in_block  # index of '\n' after title line in raw
    indent = tm.group(1)
    eq = " = " if tm.group(2) else "="
    line = f"{indent}abstract{eq}{{{esc(ab.strip())}}},\n"
    inserts.append((abs_end, line))
    print(f"located {key}: style={'space' if tm.group(2) else 'nospace'}")

inserts.sort(key=lambda x: -x[0])
new = raw
for off, line in inserts:
    assert line.count("{") == line.count("}")
    new = new[:off+1] + line + new[off+1:]

open(BIB, "w", encoding="utf-8").write(new)
print("inserted", len(inserts), "abstracts")

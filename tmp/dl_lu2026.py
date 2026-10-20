# -*- coding: utf-8 -*-
import urllib.request, ssl, os
ctx = ssl._create_unverified_context()
out = open(r"E:\wangywust.github.io\tmp\dl_lu2026_out.txt", "w", encoding="utf-8")
def log(*a): print(*a, file=out)

dest = r"E:\wangywust.github.io\pdfs\lu2026towards.pdf"
tmp = dest + ".tmp"

urls = [
    "https://dl.acm.org/doi/pdf/10.1145/3774904.3792516",
    "https://dl.acm.org/doi/pdf/10.1145/3774904.3792516?download=true",
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        body = urllib.request.urlopen(req, timeout=60, context=ctx).read()
        log(f"URL {url}: {len(body)} bytes, header={body[:20]!r}")
        if body[:5] == b"%PDF-" and len(body) > 20000:
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, dest)
            log(f"  SAVED to {dest} ({len(body)} bytes)")
            break
        else:
            log("  not a valid PDF, skipping")
    except Exception as e:
        log(f"URL {url}: ERR {e}")

out.close()

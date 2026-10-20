# -*- coding: utf-8 -*-
import urllib.request, ssl, time, json, traceback
ctx = ssl._create_unverified_context()
UA = {"User-Agent": "Mozilla/5.0"}
out = open(r"E:\wangywust.github.io\tmp\find_lu2026_out.txt", "w", encoding="utf-8")
def log(*a): print(*a, file=out)

def get(url, t=40):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=t, context=ctx).read()

time.sleep(15)  # back off S2 rate limit

# S2 by DOI
try:
    body = get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3774904.3792516?fields=title,openAccessPdf,externalIds,venue,year").decode("utf-8","ignore")
    log("S2 by DOI:", body)
except Exception:
    log("S2 DOI ERR", traceback.format_exc())

out.close()

"""Download publication PDFs using the same names used by publications.md.

The website renders PDF links as:
    https://wangywust.github.io/pdfs/{BibTeX ID}.pdf

This script mirrors that rule locally and saves each paper to:
    <repository>/pdfs/{BibTeX ID}.pdf
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from common import DATA, ROOT


REPOSITORY = ROOT.parent
DEFAULT_BIB = DATA / "citations.bib"
DEFAULT_PDF_DIR = REPOSITORY / "pdfs"
DEFAULT_MANIFEST_NAME = "paper_pdf_manifest.csv"

PDF_MAGIC = b"%PDF"
HTML_DISCOVERY_LIMIT = 1_000_000
USER_AGENT = "publication-pdf-downloader (https://wangywust.github.io)"

ACL_DOI_RE = re.compile(r"^10\.18653/v1/(?P<anthology_id>[^/#?]+)$", re.I)
ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?:v\d+)?", re.I)
ARXIV_RE = re.compile(r"(?:arxiv[:\s/]*|abs/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


@dataclass(frozen=True)
class Candidate:
    url: str
    source: str


@dataclass
class Row:
    citekey: str
    title: str
    source_url: str
    pdf_url: str
    filename: str
    status: str
    message: str


def clean_value(value: str | None) -> str:
    return (value or "").strip().strip("{}").replace("\\_", "_")


def is_placeholder(value: str) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return lowered in {"#", "todo", "tbd", "none"} or "xxxx" in lowered


def normalize_doi(value: str | None) -> str:
    value = clean_value(value)
    if not value:
        return ""
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
    return unquote(value).strip()


def arxiv_pdf_url(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip().removesuffix(".pdf")
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def candidate_from_arxiv_url(url: str) -> Candidate | None:
    parsed = urlparse(url)
    if "arxiv.org" not in parsed.netloc.lower():
        return None

    match = ARXIV_ID_RE.search(parsed.path)
    if not match:
        return None
    return Candidate(arxiv_pdf_url(match.group("id")), "arXiv")


def candidate_from_openreview_url(url: str) -> Candidate | None:
    parsed = urlparse(url)
    if "openreview.net" not in parsed.netloc.lower():
        return None

    query_id = parse_qs(parsed.query).get("id", [""])[0]
    if is_placeholder(query_id):
        return None
    if parsed.path.rstrip("/") == "/pdf":
        return Candidate(url, "OpenReview")
    return Candidate(f"https://openreview.net/pdf?id={query_id}", "OpenReview")


def candidate_from_acl_url(url: str) -> Candidate | None:
    parsed = urlparse(url)
    if "aclanthology.org" not in parsed.netloc.lower():
        return None

    paper_id = parsed.path.strip("/")
    if not paper_id or paper_id.lower().endswith(".bib"):
        return None
    if paper_id.lower().endswith(".pdf"):
        return Candidate(url, "ACL Anthology")
    return Candidate(f"https://aclanthology.org/{paper_id}.pdf", "ACL Anthology")


def candidate_from_doi(doi: str) -> Candidate | None:
    doi = normalize_doi(doi)
    if is_placeholder(doi):
        return None

    acl_match = ACL_DOI_RE.match(doi)
    if acl_match:
        paper_id = acl_match.group("anthology_id")
        return Candidate(f"https://aclanthology.org/{paper_id}.pdf", "ACL Anthology DOI")
    return Candidate(f"https://doi.org/{doi}", "DOI")


def candidate_from_url(url: str) -> list[Candidate]:
    url = clean_value(url)
    if is_placeholder(url):
        return []

    candidates = []
    for builder in (
        candidate_from_arxiv_url,
        candidate_from_openreview_url,
        candidate_from_acl_url,
    ):
        candidate = builder(url)
        if candidate:
            candidates.append(candidate)

    if url.lower().endswith(".pdf"):
        candidates.append(Candidate(url, "direct PDF"))

    parsed = urlparse(url)
    if parsed.netloc.lower() in {"doi.org", "dx.doi.org"}:
        candidate = candidate_from_doi(parsed.path.strip("/"))
        if candidate:
            candidates.append(candidate)

    candidates.append(Candidate(url, "URL"))
    return dedupe_candidates(candidates)


def candidates_for_entry(entry: dict) -> list[Candidate]:
    candidates: list[Candidate] = []
    candidates.extend(candidate_from_url(entry.get("url", "")))

    doi_candidate = candidate_from_doi(entry.get("doi", ""))
    if doi_candidate:
        candidates.append(doi_candidate)

    eprint = clean_value(entry.get("eprint", ""))
    if eprint and (
        clean_value(entry.get("archiveprefix", "")).lower() == "arxiv"
        or ARXIV_ID_RE.fullmatch(eprint)
    ):
        candidates.append(Candidate(arxiv_pdf_url(eprint), "BibTeX arXiv eprint"))

    for field in ("journal", "volume", "note", "howpublished", "booktitle"):
        match = ARXIV_RE.search(entry.get(field, ""))
        if match:
            candidates.append(Candidate(arxiv_pdf_url(match.group(1)), f"BibTeX {field} arXiv"))

    return dedupe_candidates(candidates)


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen = set()
    unique = []
    for candidate in candidates:
        if not candidate.url or candidate.url in seen:
            continue
        seen.add(candidate.url)
        unique.append(candidate)
    return unique


def public_and_local_pdf(entry: dict, pdf_dir: Path) -> tuple[str, Path]:
    public_url = f"https://wangywust.github.io/pdfs/{entry.get('ID', 'default')}.pdf"
    filename = Path(urlparse(public_url).path).name or f"{entry.get('ID', 'default')}.pdf"
    return public_url, pdf_dir / filename


def extract_pdf_links(text: str, base_url: str) -> list[Candidate]:
    links: list[Candidate] = []
    decoded = html.unescape(text)

    meta_patterns = [
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
    ]
    for pattern in meta_patterns:
        for match in re.finditer(pattern, decoded, flags=re.I):
            links.append(Candidate(urljoin(base_url, match.group(1)), "HTML citation_pdf_url"))

    for match in re.finditer(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', decoded, flags=re.I):
        links.append(Candidate(urljoin(base_url, match.group(1)), "HTML PDF link"))

    return dedupe_candidates(links)


def download_candidate(
    candidate: Candidate,
    target: Path,
    timeout: float,
) -> tuple[bool, list[Candidate], str]:
    try:
        request = urllib.request.Request(candidate.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "").lower()
            first_chunk = response.read(1024 * 64)

            if first_chunk.startswith(PDF_MAGIC) or "application/pdf" in content_type:
                target.parent.mkdir(parents=True, exist_ok=True)
                temp_target = target.with_suffix(target.suffix + ".tmp")
                with temp_target.open("wb") as handle:
                    handle.write(first_chunk)
                    while True:
                        chunk = response.read(1024 * 64)
                        if not chunk:
                            break
                        if chunk:
                            handle.write(chunk)

                with temp_target.open("rb") as handle:
                    if handle.read(4) != PDF_MAGIC:
                        temp_target.unlink(missing_ok=True)
                        return False, [], "Downloaded content is not a PDF"

                temp_target.replace(target)
                return True, [], f"Downloaded from {candidate.source}"

            if "html" in content_type or first_chunk.lstrip().startswith((b"<", b"<!")):
                body = bytearray(first_chunk)
                while len(body) < HTML_DISCOVERY_LIMIT:
                    chunk = response.read(1024 * 64)
                    if not chunk:
                        break
                    body.extend(chunk)
                    if len(body) >= HTML_DISCOVERY_LIMIT:
                        break
                text = body.decode("utf-8", errors="ignore")
                discovered = extract_pdf_links(text, response.geturl())
                if discovered:
                    return False, discovered, f"Found {len(discovered)} PDF link(s) on HTML page"

            return False, [], f"Not a PDF ({content_type or 'unknown content type'})"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, [], str(exc)


def download_entry_pdf(
    entry: dict,
    target: Path,
    timeout: float,
    sleep_seconds: float,
) -> tuple[str, str, str]:
    candidates = candidates_for_entry(entry)
    if not candidates:
        return "", "not_found", "No URL, DOI, arXiv ID, or known PDF source found"

    queue = candidates[:]
    tried = set()
    last_message = ""
    first_pdf_url = queue[0].url

    while queue:
        candidate = queue.pop(0)
        if candidate.url in tried:
            continue
        tried.add(candidate.url)

        ok, discovered, message = download_candidate(candidate, target, timeout)
        if ok:
            return candidate.url, "downloaded", message

        last_message = f"{candidate.source}: {message}"
        queue.extend(discovered)
        if sleep_seconds:
            time.sleep(sleep_seconds)

    return first_pdf_url, "download_failed", last_message or "No candidate downloaded successfully"


def find_entry_end(text: str, open_brace: int) -> int:
    depth = 1
    index = open_brace + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("Unclosed BibTeX entry")


def split_key_and_fields(body: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(body):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            return body[:index].strip(), body[index + 1 :]
    return body.strip(), ""


def read_bib_value(text: str, start: int) -> tuple[str, int]:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1

    if index >= len(text):
        return "", index

    if text[index] == "{":
        depth = 1
        index += 1
        value_start = index
        while index < len(text):
            char = text[index]
            if char == "\\":
                index += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[value_start:index].strip(), index + 1
            index += 1
        return text[value_start:].strip(), index

    if text[index] == '"':
        index += 1
        value_start = index
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == '"':
                return text[value_start:index].strip(), index + 1
            index += 1
        return text[value_start:].strip(), index

    value_start = index
    while index < len(text) and text[index] != ",":
        index += 1
    return text[value_start:index].strip(), index


def parse_fields(text: str) -> dict:
    fields = {}
    index = 0
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] == ","):
            index += 1
        if index >= len(text):
            break

        name_start = index
        while index < len(text) and (text[index].isalnum() or text[index] in "_-"):
            index += 1
        name = text[name_start:index].strip().lower()

        while index < len(text) and text[index].isspace():
            index += 1
        if not name or index >= len(text) or text[index] != "=":
            break

        value, index = read_bib_value(text, index + 1)
        fields[name] = value
    return fields


def load_entries(bib_path: Path) -> list[dict]:
    text = bib_path.read_text(encoding="utf-8")
    entries: list[dict] = []
    index = 0
    while True:
        at_index = text.find("@", index)
        if at_index == -1:
            break
        open_brace = text.find("{", at_index)
        if open_brace == -1:
            break

        entry_type = text[at_index + 1 : open_brace].strip().lower()
        if entry_type in {"comment", "preamble", "string"}:
            index = open_brace + 1
            continue

        close_brace = find_entry_end(text, open_brace)
        body = text[open_brace + 1 : close_brace]
        citekey, fields_text = split_key_and_fields(body)
        if citekey:
            entry = parse_fields(fields_text)
            entry["ID"] = citekey
            entry["ENTRYTYPE"] = entry_type
            entries.append(entry)
        index = close_brace + 1
    return entries


def write_manifest(rows: list[Row], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["citekey", "title", "source_url", "pdf_url", "filename", "status", "message"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    temp_path.replace(manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDFs to the same pdfs/{BibTeX ID}.pdf paths used by publications.md."
    )
    parser.add_argument("bib_path", nargs="?", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ids", nargs="+", help="Only process selected BibTeX IDs.")
    parser.add_argument("--limit", type=int, help="Process at most this many entries after filtering.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to wait between download attempts.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    entries = load_entries(args.bib_path)

    if args.ids:
        wanted = set(args.ids)
        entries = [entry for entry in entries if entry.get("ID") in wanted]
    if args.limit:
        entries = entries[: args.limit]

    rows: list[Row] = []
    for entry in entries:
        citekey = entry.get("ID", "default")
        public_url, target = public_and_local_pdf(entry, args.pdf_dir)
        source_url = clean_value(entry.get("url", ""))
        title = clean_value(entry.get("title", ""))

        if target.exists() and not args.overwrite:
            status, message, candidate_url = "exists", "PDF already exists", ""
        elif args.dry_run:
            candidates = candidates_for_entry(entry)
            status = "would_download" if candidates else "not_found"
            message = "Dry run; no files written" if candidates else "No download candidate found"
            candidate_url = candidates[0].url if candidates else ""
        else:
            candidate_url, status, message = download_entry_pdf(
                entry=entry,
                target=target,
                timeout=args.timeout,
                sleep_seconds=args.sleep,
            )

        rows.append(
            Row(
                citekey=citekey,
                title=title,
                source_url=source_url,
                pdf_url=candidate_url or public_url,
                filename=target.name,
                status=status,
                message=message,
            )
        )
        line = f"[{status:15}] {citekey}: {target}"
        if candidate_url and candidate_url != public_url:
            line += f" <- {candidate_url}"
        print(line)

    if not args.dry_run and not args.no_manifest:
        manifest = args.manifest or (args.pdf_dir / DEFAULT_MANIFEST_NAME)
        write_manifest(rows, manifest)
        print(f"\nManifest written to {manifest}")


if __name__ == "__main__":
    main()

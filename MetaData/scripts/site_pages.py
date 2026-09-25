"""Render the repository's Jekyll pages from resume metadata."""

from __future__ import annotations

from common import ROOT, load, value


REPOSITORY = ROOT.parent
START = "<!-- METADATA:{name}:START -->"
END = "<!-- METADATA:{name}:END -->"


def md_item(item: dict, language: str) -> str:
    primary = value(item, "primary", language)
    url = item.get("url")
    label = f"[{primary}]({url})" if url else primary
    secondary = value(item, "secondary", language)
    date = value(item, "date", language)
    suffix = ", ".join(part for part in (secondary, date) if part)
    return f"- {label}" + (f" — {suffix}" if suffix else "")


def section(name: str, language: str, heading_level: int = 3) -> str:
    data = load(name)
    title = value(data, "title", language)
    lines = [f"{'#' * heading_level} {title}", ""]
    lines.extend(md_item(item, language) for item in data["items"])
    return "\n".join(lines)


def write_page(filename: str, layout: str, body: str) -> None:
    text = f"---\nlayout: {layout}\n---\n\n{body.rstrip()}\n"
    (REPOSITORY / filename).write_text(text, encoding="utf-8")


def replace_managed_block(text: str, name: str, body: str) -> str:
    start = START.format(name=name)
    end = END.format(name=name)
    replacement = f"{start}\n{body.rstrip()}\n{end}"
    if start not in text or end not in text:
        raise ValueError(f"Missing managed markers for {name}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + replacement + after


def render_site_pages() -> None:
    services = "\n\n".join([
        section("editorship", "en"),
        section("area_chair", "en"),
        section("reviewing", "en"),
    ])
    write_page("services.md", "course", services)

    write_page("awards.md", "course", section("awards", "en", 2))

    teaching = "\n\n".join([
        section("tutorial", "en", 2),
        section("teaching", "en", 2),
    ])
    write_page("teaching.md", "course", teaching)

    index_path = REPOSITORY / "index.md"
    index = index_path.read_text(encoding="utf-8")
    index = replace_managed_block(index, "work", section("work", "zh", 2))
    index = replace_managed_block(index, "education", section("education", "zh", 2))
    index_path.write_text(index, encoding="utf-8")


if __name__ == "__main__":
    render_site_pages()

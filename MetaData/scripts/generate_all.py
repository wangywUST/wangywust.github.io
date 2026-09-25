"""Generate every TeX fragment and the assembled Markdown resumes."""

from common import GENERATED, render_profile, render_section
from publications_latex import generate_latex_publications
from publications_markdown import generate_markdown_publications
from site_pages import render_site_pages


SECTIONS = ("work", "education", "area_chair", "editorship", "reviewing", "tutorial", "grants", "awards", "teaching")


def main() -> None:
    generate_latex_publications()
    generate_markdown_publications()
    render_profile()
    for section in SECTIONS:
        render_section(section)
    for language in ("zh", "en"):
        parts = [(GENERATED / language / "profile.md").read_text(encoding="utf-8").rstrip()]
        for section in SECTIONS:
            parts.append((GENERATED / language / f"{section}.md").read_text(encoding="utf-8").rstrip())
        (GENERATED / f"resume_{language}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    render_site_pages()


if __name__ == "__main__":
    main()

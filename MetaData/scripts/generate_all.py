"""Generate every TeX fragment and the assembled Markdown resumes."""

from common import GENERATED, render_profile, render_section


SECTIONS = ("work", "education", "honors", "teaching", "tutorial", "editorship", "area_chair", "reviewing")


def main() -> None:
    render_profile()
    for section in SECTIONS:
        render_section(section)
    for language in ("zh", "en"):
        parts = [(GENERATED / language / "profile.md").read_text(encoding="utf-8").rstrip()]
        for section in SECTIONS:
            parts.append((GENERATED / language / f"{section}.md").read_text(encoding="utf-8").rstrip())
        (GENERATED / f"resume_{language}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

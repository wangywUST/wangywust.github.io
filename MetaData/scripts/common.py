"""Shared renderers for the resume metadata pipeline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GENERATED = ROOT / "generated"


def load(section: str) -> dict:
    with (DATA / f"{section}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def write(section: str, language: str, extension: str, text: str) -> Path:
    target = GENERATED / language / f"{section}.{extension}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")
    return target


def value(item: dict, field: str, language: str) -> str:
    return item.get(f"{field}_{language}", item.get(field, ""))


def tex_escape(text: str) -> str:
    """Escape metadata characters that have a special meaning in LaTeX."""
    return text.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def latex_rating_labels(ratings: list[str]) -> str:
    """Render metadata ratings in the same red style as publications."""
    return " ".join(f"\\textcolor{{red}}{{({rating})}}" for rating in ratings)


def render_section(section: str) -> list[Path]:
    """Render a conventional resume section to zh/en TeX and Markdown."""
    data = load(section)
    outputs: list[Path] = []
    for language in ("zh", "en"):
        title = value(data, "title", language)
        bold = data.get(f"bold_{language}", data.get("bold", True))
        tex_lines = ([f"\\rSection{{{title}}}", "\\begin{itemize}"] if language == "zh"
                     else [f"\\begin{{rSection}}{{{title}}}"])
        md_lines = [f"## {title}", ""]

        for item in data["items"]:
            primary = value(item, "primary", language)
            secondary = value(item, "secondary", language)
            date = value(item, "date", language)
            url = value(item, "url", language)
            primary_tex = tex_escape(primary)
            secondary_tex = tex_escape(secondary)
            date_tex = tex_escape(date)
            if url:
                primary_tex = f"\\href{{{tex_escape(url)}}}{{{primary_tex}}}"
            ratings_tex = latex_rating_labels(item.get("ratings", []))
            if language == "zh":
                formatted_primary = f"\\textbf{{{primary_tex}}}" if bold else primary_tex
                line = f"  \\item {formatted_primary}"
                if date:
                    line += f" \\hfill {date_tex}"
                if secondary:
                    line += f"\\\\\n  {secondary_tex}"
                if ratings_tex:
                    line += f" {ratings_tex}"
                tex_lines.append(line)
            else:
                line = f"{{\\bf {primary_tex}}}" if bold else primary_tex
                if date:
                    line += f" \\hfill {{{date_tex}}}"
                if secondary:
                    line += f"\\\\\n{secondary_tex}"
                if ratings_tex:
                    line += f" {ratings_tex}"
                tex_lines.extend([line, ""])

            primary_md = f"[{primary}]({url})" if url else primary
            md = f"- **{primary_md}**"
            if date:
                md += f" ({date})"
            if secondary:
                md += f" — {secondary}"
            md_lines.append(md)

        tex_lines.append("\\end{itemize}" if language == "zh" else "\\end{rSection}")
        outputs.append(write(section, language, "tex", "\n".join(tex_lines)))
        outputs.append(write(section, language, "md", "\n".join(md_lines)))
    return outputs


def render_profile() -> list[Path]:
    data = load("profile")
    outputs: list[Path] = []
    for language in ("zh", "en"):
        name = value(data, "name", language)
        email = data["email"]
        website = data["website"]
        wechat = data["wechat"]
        if language == "zh":
            tex = ("\\begin{center}\n"
                   f"    {{\\Large \\textbf{{{name}}}}}\\\\\n"
                   f"    电子邮箱： \\href{{mailto:{email}}}{{{email}}} \\quad "
                   f"微信号：{wechat} \\quad 个人主页： \\href{{{website}}}{{{website}}}\n"
                   "\\end{center}")
            md = f"# {name}\n\n邮箱：[{email}](mailto:{email}) · 微信：{wechat} · 主页：[{website}]({website})"
        else:
            tex = (f"\\name{{{name}}}\n"
                   f"\\address{{Email: {email} \\\\ Website: {website} \\\\ Wechat: {wechat}}}")
            md = f"# {name}\n\nEmail: [{email}](mailto:{email}) · WeChat: {wechat} · Website: [{website}]({website})"
        outputs.append(write("profile", language, "tex", tex))
        outputs.append(write("profile", language, "md", md))
    return outputs

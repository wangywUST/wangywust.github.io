"""Update the existing faculty information form while preserving its design.

Bindings and update policies live in data/faculty_form.json. Detailed education
and work narratives are filled only when absent. Other bound blocks are synced
from metadata; existing service roles and award-stage headings are retained.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.text.paragraph import Paragraph
from lxml import etree

from common import DATA, value
from publications_data import read_publications
from word_common import add_numbering, add_rating


@dataclass
class Entry:
    text: str
    heading: bool = False
    list_name: str | None = None
    ratings: tuple[str, ...] = ()


@dataclass
class FormUpdate:
    path: Path
    changed: bool
    backup: Path | None = None


def _normalize(text):
    return re.sub(r"[\s:：]", "", text)


def _text(element):
    return "".join(node.text or "" for node in element.iter(qn("w:t")))


def _paragraphs(rows):
    return [p for row in rows for cell in row.findall(qn("w:tc")) for p in cell.findall(qn("w:p"))]


def _year(text):
    match = re.search(r"\d{4}", str(text))
    return int(match[0]) if match else 0


def _dated(item):
    date = value(item, "date", "zh").strip()
    if re.fullmatch(r"\d{4}", date):
        date += "年"
    return f"{date}，" if date else ""


def _venue_key(text):
    text = text.replace("（", "(").replace("）", ")").split("，担任", 1)[0]
    acronyms = re.findall(r"\(([A-Za-z][A-Za-z0-9 /-]*)\)", text)
    return re.sub(r"\W", "", acronyms[-1] if acronyms else text).lower()


class FormRenderer:
    def __init__(self, document, metadata, config, publications_source):
        self.document, self.metadata, self.config = document, metadata, config
        self.publications_source = publications_source
        self.bindings = config["sections"]
        self.aliases = {_normalize(alias): key for key, binding in self.bindings.items() for alias in binding["aliases"]}
        candidates = [(sum(self.key(row) is not None for row in table._tbl.findall(qn("w:tr"))), table) for table in document.tables]
        if not candidates or max(score for score, _ in candidates) == 0:
            raise ValueError("未找到教师信息表的栏目标题，请检查 Word 表格或 faculty_form.json 中的 aliases。")
        self.table = max(candidates, key=lambda candidate: candidate[0])[1]
        initial = self.sections()
        self.header_template = deepcopy(next(iter(initial.values()))[0])
        body_rows = initial.get("grants", (None, []))[1]
        if not body_rows:
            body_rows = [row for row in self.table._tbl.findall(qn("w:tr")) if not self.is_header(row) and len(row.findall(qn("w:tc"))) == 1]
        if not body_rows:
            raise ValueError("表格缺少可复用的正文行，无法保留原有版式。")
        self.row_template = deepcopy(body_rows[0])
        self.p_template = deepcopy(next(p for p in _paragraphs(body_rows) if _text(p).strip()))
        self.run_template = deepcopy(next(r for r in self.p_template.findall(qn("w:r")) if _text(r).strip()))
        # Normalize only layout controls that this renderer owns. Font, size,
        # boldness, table width, and cell padding come from the existing form.
        properties = self.p_template.get_or_add_pPr()
        for tag in ("pStyle", "numPr", "ind", "tabs", "keepNext", "keepLines", "spacing"):
            for child in properties.findall(qn(f"w:{tag}")):
                properties.remove(child)
        for child in list(self.p_template):
            if child.tag != qn("w:pPr"):
                self.p_template.remove(child)
        for child in list(self.run_template):
            if child.tag != qn("w:rPr"):
                self.run_template.remove(child)

    def key(self, row):
        return self.aliases.get(_normalize(_text(row)))

    def is_header(self, row):
        if self.key(row) is not None:
            return True
        cells = row.findall(qn("w:tc"))
        if len(cells) != 1 or len(_text(row)) > 60:
            return False
        shading = cells[0].find('./' + qn('w:tcPr') + '/' + qn('w:shd'))
        return shading is not None and shading.get(qn("w:fill"), "FFFFFF").upper() not in ("FFFFFF", "AUTO", "NONE")

    def sections(self):
        sections, active = {}, None
        for row in self.table._tbl.findall(qn("w:tr")):
            if self.is_header(row):
                active = self.key(row)
                if active is not None:
                    if active in sections:
                        raise ValueError(f"栏目标题重复：{active}，为避免覆盖错误已停止。")
                    sections[active] = (row, [])
            elif active is not None:
                sections[active][1].append(row)
        return sections

    def ensure_headers(self):
        keys = list(self.bindings)
        for index, key in enumerate(keys):
            sections = self.sections()
            if key in sections:
                continue
            row = deepcopy(self.header_template)
            cell = row.find(qn("w:tc"))
            paragraph = cell.find(qn("w:p"))
            run = deepcopy(next(paragraph.iter(qn("w:r"))))
            for child in list(run):
                if child.tag != qn("w:rPr"):
                    run.remove(child)
            node = OxmlElement("w:t")
            node.text = self.bindings[key]["heading"]
            run.append(node)
            for child in list(cell):
                if child.tag != qn("w:tcPr"):
                    cell.remove(child)
            for child in list(paragraph):
                if child.tag != qn("w:pPr"):
                    paragraph.remove(child)
            paragraph.append(run)
            cell.append(paragraph)
            later = next((sections[k][0] for k in keys[index + 1:] if k in sections), None)
            if later is None:
                self.table._tbl.append(row)
            else:
                later.addprevious(row)

    def paragraph(self, entry):
        element = deepcopy(self.p_template)
        paragraph = Paragraph(element, self.document._body)
        paragraph.paragraph_format.keep_together = True
        paragraph.paragraph_format.keep_with_next = entry.heading
        if entry.heading:
            paragraph.paragraph_format.space_before = Pt(6)
        if entry.list_name:
            number_id = add_numbering(self.document, format_text="[%1]" if entry.list_name == "papers" else "%1.", name=f"FacultyForm:{entry.list_name}")
            number = element.get_or_add_pPr().get_or_add_numPr()
            number.get_or_add_ilvl().val = 0
            number.get_or_add_numId().val = number_id
        run = deepcopy(self.run_template)
        node = OxmlElement("w:t")
        node.text = entry.text
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        run.append(node)
        element.append(run)
        for rating in entry.ratings:
            before = len(paragraph.runs)
            add_rating(paragraph, rating)
            label = paragraph.runs[before]
            if self.run_template.rPr is not None:
                old = label._r.rPr
                if old is not None:
                    label._r.remove(old)
                label._r.insert(0, deepcopy(self.run_template.rPr))
                fonts = label._r.rPr.find(qn("w:rFonts"))
                if fonts is not None:
                    fonts.attrib.pop(qn("w:hint"), None)
            label.font.color.rgb = RGBColor(255, 0, 0)
        return element

    def replace(self, key, entries):
        header, old_rows = self.sections()[key]
        for row in old_rows:
            row.getparent().remove(row)
        # A subheading shares a row with its first record, avoiding orphaned
        # headings without turning a whole section into one unbreakable row.
        groups, pending = [], []
        for entry in entries:
            pending.append(entry)
            if not entry.heading:
                groups.append(pending)
                pending = []
        if pending:
            groups.append(pending)
        previous = header
        for index, group in enumerate(groups):
            row = deepcopy(self.row_template)
            properties = row.get_or_add_trPr()
            for child in list(properties):
                properties.remove(child)
            properties.append(OxmlElement("w:cantSplit"))
            cell = row.find(qn("w:tc"))
            for child in list(cell):
                if child.tag != qn("w:tcPr"):
                    cell.remove(child)
            borders = cell.get_or_add_tcPr().find(qn("w:tcBorders"))
            if borders is None:
                borders = OxmlElement("w:tcBorders")
                cell.get_or_add_tcPr().append(borders)
            for edge in ("top", "bottom"):
                border = borders.find(qn(f"w:{edge}"))
                if border is None:
                    border = OxmlElement(f"w:{edge}")
                    borders.append(border)
                border.attrib.clear()
                border.set(qn("w:val"), "single" if edge == "bottom" and index == len(groups) - 1 else "nil")
                if border.get(qn("w:val")) == "single":
                    border.set(qn("w:sz"), "4")
                    border.set(qn("w:color"), "000000")
            for entry in group:
                cell.append(self.paragraph(entry))
            previous.addnext(row)
            previous = row

    def biography(self, key):
        name = self.metadata["profile"]["name_zh"]
        sentences = []
        for item in sorted(self.metadata[key]["items"], key=lambda item: _year(value(item, "date", "zh"))):
            primary = value(item, "primary", "zh")
            institution, _, position = primary.partition("，")
            if not position:
                institution, _, position = primary.partition(",")
            date = re.sub(r"\s*-\s*", "至", value(item, "date", "zh"))
            if key == "education":
                sentence = f"{date}，{name}在{institution.strip()}攻读{position.strip()}学位。"
            else:
                department = value(item, "secondary", "zh")
                sentence = f"{date}，{name}在{institution.strip()}{department}担任{position.strip()}。"
            sentences.append(sentence)
        return [Entry("".join(sentences))]

    def teaching(self):
        items = self.metadata["teaching"]["items"]
        groups = [("主讲课程：", [i for i in items if "助教" not in value(i, "primary", "zh")]),
                  ("助教经历：", [i for i in items if "助教" in value(i, "primary", "zh")])]
        entries = []
        for title, records in groups:
            if records:
                entries.append(Entry(title, heading=True))
                entries.extend(Entry(_dated(item) + value(item, "primary", "zh").rstrip("。") + "。") for item in records)
        return entries

    def grants(self):
        return [Entry(_dated(item) + value(item, "primary", "zh").rstrip("。") + "。", list_name="grants") for item in self.metadata["grants"]["items"]]

    def awards(self, old_rows):
        order, existing, stage = [], {}, None
        for p in _paragraphs(old_rows):
            text = _text(p).strip()
            if re.match(r"^.{1,12}期间[:：]$", text):
                stage = text.rstrip(":：")
                if stage not in order:
                    order.append(stage)
            elif stage and _year(text):
                existing[_year(text)] = stage
        ranges = []
        for key in ("education", "work"):
            for item in self.metadata[key]["items"]:
                label = value(item, "primary", "zh")
                group = next((title for word, title in (("博士后", "博士后期间"), ("教授", "担任教职期间"), ("本科", "本科期间"), ("硕士", "硕士期间"), ("博士", "博士期间")) if word in label), None)
                years = re.findall(r"\d{4}", value(item, "date", "zh"))
                if group and years:
                    ranges.append((int(years[0]), int(years[-1]) if len(years) > 1 else 9999, group))
        grouped = {}
        for item in self.metadata["awards"]["items"]:
            year = _year(item.get("date", ""))
            candidates = [r for r in ranges if r[0] <= year <= r[1]]
            stage = existing.get(year) or (max(candidates)[2] if candidates else "其他奖项")
            grouped.setdefault(stage, []).append(item)
        entries = []
        for stage in order + [s for s in grouped if s not in order]:
            if stage not in grouped:
                continue
            entries.append(Entry(stage + "：", heading=True))
            for item in sorted(grouped[stage], key=lambda item: -_year(item.get("date", ""))):
                title = value(item, "primary", "zh")
                english = value(item, "primary", "en")
                if "Best Paper Finalist" in english:
                    title = re.sub(r"最佳论文.*", "最佳论文入围奖（Best Paper Finalist）", title)
                elif "Best Paper" in english and "最佳论文" in title:
                    title = title.replace("Workshop最佳论文", "Workshop 最佳论文奖")
                entries.append(Entry(_dated(item) + title.rstrip("。") + "。", list_name="awards:" + stage))
        return entries

    def activities(self, old_rows):
        roles, titles, extra = {}, {}, []
        aliases = {"期刊编委": "editorship", "会议领域主席": "area_chair", "会议主席": "area_chair", "会议（高级）领域主席": "area_chair", "学术会议Tutorial": "tutorial", "会议审稿": "conference_review", "期刊审稿": "journal_review"}
        unknown = False
        for p in _paragraphs(old_rows):
            text = _text(p).strip()
            if not text:
                continue
            normalized = _normalize(text)
            if normalized in aliases:
                titles[aliases[normalized]] = text
                unknown = False
            elif len(text) < 50 and text.endswith((":", "：")):
                unknown = True
            if unknown:
                extra.append(Entry(text, heading=text.endswith((":", "："))))
            match = re.search(r"担任\s*(Senior Area Chair|Area Chair|Action Editor|Associate Editor)", text)
            if match:
                roles[_venue_key(text)] = match[1]
        entries = []
        for section, default_title, default_role in (("editorship", "期刊编委：", "Associate Editor"), ("area_chair", "会议领域主席：", "Area Chair")):
            entries.append(Entry(titles.get(section, default_title), heading=True))
            for item in self.metadata[section]["items"]:
                label = value(item, "primary", "zh")
                role = item.get("role") or roles.get(_venue_key(label), default_role)
                ratings = tuple(f"({rating})" for rating in item.get("ratings", []))
                entries.append(Entry(f"{label}，担任 {role}。", ratings=ratings))
        entries.append(Entry(titles.get("tutorial", "学术会议 Tutorial："), heading=True))
        for item in self.metadata["tutorial"]["items"]:
            entries.append(Entry(f'{item["date"]}，主办“{value(item, "primary", "zh")}”（{value(item, "primary", "en")}）。'))
        for item in self.metadata["reviewing"]["items"]:
            primary = value(item, "primary", "zh")
            venues = [venue.strip() for venue in re.split(r"[;；]", value(item, "secondary", "zh")) if venue.strip()]
            if "会议" in primary:
                entries.extend([Entry(titles.get("conference_review", "会议审稿："), heading=True), Entry("、".join(venues) + "，担任审稿人。")])
            else:
                entries.append(Entry(titles.get("journal_review", "期刊审稿："), heading=True))
                entries.extend(Entry(venue + "，担任审稿人。") for venue in venues)
        entries.extend(extra)
        return entries

    def render(self, publications_mode):
        self.ensure_headers()
        for key, binding in self.bindings.items():
            header, rows = self.sections()[key]
            nonempty = any(_text(row).strip() for row in rows)
            if binding["update"] == "fill" and nonempty:
                continue
            if key in ("education", "work"):
                entries = self.biography(key)
            elif key == "awards":
                entries = self.awards(rows)
            elif key == "activities":
                entries = self.activities(rows)
            elif key == "publications":
                if publications_mode == "link":
                    url = self.metadata["profile"]["website"].rstrip("/") + "/publications"
                    if nonempty and url in "".join(_text(row) for row in rows) and len(_paragraphs(rows)) == 1:
                        continue
                    entries = [Entry("完整论文列表请见：" + url)]
                else:
                    entries = []
                    for paper in read_publications(self.publications_source):
                        separator = "" if paper.title.endswith((".", "?", "!")) else "."
                        entries.append(Entry(f"{paper.authors} {paper.title}{separator} {paper.venue}", list_name="papers", ratings=paper.ratings))
            else:
                entries = getattr(self, key)()
            Paragraph(header.find('./' + qn('w:tc') + '/' + qn('w:p')), self.document._body).paragraph_format.keep_with_next = True
            self.replace(key, entries)


def _canonical(xml):
    return etree.tostring(etree.fromstring(xml), method="c14n")


def update_faculty_form(target=None, *, data_dir=DATA, publications_source=None, publications_mode=None, backups=True) -> FormUpdate:
    data_dir = Path(data_dir)
    project = data_dir.parent
    config = json.loads((data_dir / "faculty_form.json").read_text(encoding="utf-8"))
    target = Path(target) if target is not None else project / config["document"]
    destination = target
    mode = publications_mode or config["publications_mode"]
    if mode not in ("link", "full"):
        raise ValueError("publications_mode must be link or full")
    source = Path(publications_source) if publications_source is not None else project / "generated/publications.md"
    metadata = {name: json.loads((data_dir / f"{name}.json").read_text(encoding="utf-8")) for name in ("profile", "education", "work", "teaching", "grants", "awards", "editorship", "area_chair", "tutorial", "reviewing")}
    original = target.read_bytes()
    document = Document(BytesIO(original))
    FormRenderer(document, metadata, config, source).render(mode)
    replacements = {
        "word/document.xml": etree.tostring(document.element, encoding="UTF-8", xml_declaration=True, standalone=True),
        "word/numbering.xml": etree.tostring(document.part.numbering_part.element, encoding="UTF-8", xml_declaration=True, standalone=True),
    }
    with ZipFile(BytesIO(original)) as package:
        changed = any(_canonical(package.read(name)) != _canonical(content) for name, content in replacements.items())
    if not changed:
        print(f"Faculty form is already up to date: {target}")
        return FormUpdate(target, False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if backups:
        folder = project / config["backups_directory"]
        folder.mkdir(parents=True, exist_ok=True)
        backup = folder / f'{target.stem}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.zip'
        with ZipFile(backup, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(target.name, original)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", prefix=".faculty_form_", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(BytesIO(original)) as source_package, ZipFile(temporary_path, "w") as result:
            for member in source_package.infolist():
                result.writestr(member, replacements.get(member.filename, source_package.read(member.filename)))
        if target.read_bytes() != original:
            raise RuntimeError("Word 文件在生成期间被修改；为保留刚保存的内容，请重新运行。")
        try:
            os.replace(temporary_path, destination)
        except PermissionError as error:
            raise PermissionError(f"请保存并关闭 Word 中的 {target.name} 后重新运行；原文件未改动，未生成 Word 副本。") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Faculty form updated: {destination}")
    return FormUpdate(destination, changed, backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, help="Existing Word form to update")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--publications", choices=("link", "full"), default=None)
    parser.add_argument("--no-backup", action="store_true")
    arguments = parser.parse_args()
    update_faculty_form(arguments.document, data_dir=arguments.data_dir, publications_mode=arguments.publications, backups=not arguments.no_backup)


if __name__ == "__main__":
    main()

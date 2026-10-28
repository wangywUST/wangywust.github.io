"""Numbering and rating formatting inside the final faculty information form."""

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import RGBColor


def _element(tag: str, **attributes):
    element = OxmlElement(f"w:{tag}")
    for key, value in attributes.items():
        element.set(qn(f"w:{key}"), str(value))
    return element


def add_numbering(document, format_text="[%1]", name=None) -> int:
    """Create a fresh [1], [2], ... list; separate groups restart at [1]."""
    numbering = document.part.numbering_part.element
    if name is not None:
        for definition in numbering.findall(qn("w:abstractNum")):
            label = definition.find(qn("w:name"))
            if label is not None and label.get(qn("w:val")) == name:
                abstract_id = definition.get(qn("w:abstractNumId"))
                for number in numbering.findall(qn("w:num")):
                    if number.find(qn("w:abstractNumId")).get(qn("w:val")) == abstract_id:
                        return int(number.get(qn("w:numId")))
    abstract_id = 1 + max(int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum")))
    num_id = 1 + max(int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num")))
    abstract = _element("abstractNum", abstractNumId=abstract_id)
    abstract.append(_element("multiLevelType", val="singleLevel"))
    if name is not None:
        abstract.append(_element("name", val=name))
    level = _element("lvl", ilvl=0)
    for tag, value in (("start", "1"), ("numFmt", "decimal"), ("lvlText", format_text), ("lvlJc", "left")):
        level.append(_element(tag, val=value))
    properties = _element("pPr")
    tabs = _element("tabs")
    tabs.append(_element("tab", val="num", pos=576))
    properties.append(tabs)
    properties.append(_element("ind", left=576, hanging=576))
    level.append(properties)
    abstract.append(level)
    numbering.insert(0, abstract)
    number = _element("num", numId=num_id)
    number.append(_element("abstractNumId", val=abstract_id))
    numbering.append(number)
    return num_id


def add_rating(paragraph, rating: str) -> None:
    """Color the entire parenthesized label, including both parentheses, red."""
    run = paragraph.add_run()
    if rating == "(CCF-A)":
        run.add_text(" (CCF")
        run._r.append(_element("noBreakHyphen"))
        run.add_text("A)")
    else:
        run.add_text(" " + rating.replace("JCR Q1", "JCR\u00a0Q1"))
    run.font.color.rgb = RGBColor(255, 0, 0)



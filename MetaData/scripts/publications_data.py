"""Read the publication records used to fill the faculty information form.

The Markdown file contains HTML. Its publication order and rating labels are
the authority; this reader does not classify venues or write standalone files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lxml import html

@dataclass(frozen=True)
class Publication:
    authors: str
    title: str
    venue: str
    ratings: tuple[str, ...]


def clean_text(text: str) -> str:
    """Decode leftover LaTeX escapes and collapse HTML formatting whitespace."""
    text = re.sub(r"\\([%&_#$])", r"\1", text)
    return " ".join(text.split())


def read_publications(source: Path) -> list[Publication]:
    tree = html.fromstring(source.read_text(encoding="utf-8"))
    publications = []
    for item in tree.xpath('//li[.//div[contains(concat(" ", @class, " "), " title ")]]'):
        def field(name: str):
            matches = item.xpath(
                './/div[contains(concat(" ", normalize-space(@class), " "), $name)]',
                name=f" {name} ",
            )
            if len(matches) != 1:
                raise ValueError(f"Publication {len(publications) + 1}: expected one {name} field")
            return matches[0]

        periodical = field("periodical")
        venue_nodes = periodical.xpath(".//em")
        rating_text = clean_text(periodical.text_content())
        ratings = tuple(re.findall(r"\((?:CCF-A|JCR Q1)\)", rating_text))
        venue = clean_text(venue_nodes[0].text_content()) if venue_nodes else rating_text
        if not venue_nodes:
            for rating in ratings:
                venue = venue.replace(rating, "")
            venue = clean_text(venue)
        publication = Publication(
            authors=clean_text(field("author").text_content()),
            title=clean_text(field("title").text_content()),
            venue=venue,
            ratings=ratings,
        )
        if not all((publication.authors, publication.title, publication.venue)):
            raise ValueError(f"Publication {len(publications) + 1}: empty bibliographic field")
        publications.append(publication)
    if not publications:
        raise ValueError(f"No publication entries found in {source}")
    return publications



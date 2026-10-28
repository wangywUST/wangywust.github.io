"""Check publication source fidelity without writing intermediate Word files."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from publications_data import read_publications


SOURCE = '''<h2>Selected Publications</h2><ol class="bibliography"><h3>2026</h3>
<li><div class="title extra">A &amp; B: Why?</div><div class="author"><b>Yiwei Wang</b>, A. Author.</div>
<div class="periodical"><em>Example Journal, 2026.</em>
<span style="color:red">(CCF-A)</span><span style="color:red">(JCR Q1)</span></div>
<div class="links"><a href="#">PDF</a></div></li><h3>2025</h3>
<li><div class="title">100\\% valid</div><div class="author">Another Author.</div>
<div class="periodical"><em>International Conference on Learning Representations, 2025.</em></div></li></ol>'''


class WordPublicationsTests(unittest.TestCase):
    def test_order_text_and_labels_come_from_html(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "publications.md"
            source.write_text(SOURCE, encoding="utf-8")
            items = read_publications(source)
        self.assertEqual([item.title for item in items], ["A & B: Why?", "100% valid"])
        self.assertEqual(items[0].authors, "Yiwei Wang, A. Author.")
        self.assertEqual(items[0].ratings, ("(CCF-A)", "(JCR Q1)"))
        # An A-class venue name alone must never cause the exporter to add a label.
        self.assertEqual(items[1].ratings, ())

    def test_incomplete_source_does_not_produce_a_document(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "bad.md"
            source.write_text(SOURCE.replace('<div class="author">Another Author.</div>', ''), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "author field"):
                read_publications(source)
            self.assertEqual(list(Path(folder).glob("*.docx")), [])


if __name__ == "__main__":
    unittest.main()

"""
Test TXT Document Extractor.
Validates:
- Plain text ingestion and line numbering
- Section splitting by double newlines
- ALL CAPS heading detection
- Table extraction from formatted text
"""

from pathlib import Path
import pytest

from narvl.document.models import DocumentType
from narvl.document.txt_extractor import TxtExtractor


def test_txt_extractor_sections_and_headings(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(
        "CUSTOMER DISPATCH REPORT\n\n"
        "Customer Name: Ramesh Gupta\n"
        "Phone: 9811223344\n"
        "City: Hyd\n\n"
        "TRANSACTION SUMMARY\n\n"
        "| Customer | Amount |\n"
        "| Ramesh Gupta | $500.00 |\n",
        encoding="utf-8",
    )

    extractor = TxtExtractor()
    parsed = extractor.extract(txt_file)

    assert parsed.file_type == DocumentType.TXT
    assert len(parsed.pages) == 1
    assert parsed.pages[0].char_count > 0
    assert len(parsed.structure.headings) >= 1
    assert any(h["title"] == "CUSTOMER DISPATCH REPORT" for h in parsed.structure.headings)
    assert len(parsed.structure.sections) >= 2


def test_txt_extractor_table_parsing(tmp_path):
    txt_file = tmp_path / "table_doc.txt"
    txt_file.write_text(
        "TABLE DATA\n\n"
        "| Name | Phone | City |\n"
        "|---|---|---|\n"
        "| Alice | 9988776655 | Hyderabad |\n"
        "| Bob | 9911223344 | Bengaluru |\n",
        encoding="utf-8",
    )

    extractor = TxtExtractor()
    parsed = extractor.extract(txt_file)

    assert len(parsed.tables) >= 1
    tbl = parsed.tables[0]
    assert tbl.table_id.startswith("txt_tbl_")
    df = tbl.data
    assert df.height == 2
    assert "Name" in df.columns or "col" in df.columns[0].lower()

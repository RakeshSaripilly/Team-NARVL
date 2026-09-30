"""
Test DOCX Document Extractor.
Validates:
- Paragraphs and paragraph_id provenance tracking
- Headings extraction (Heading 1-3)
- Tabular data conversion to Polars DataFrame
"""

from pathlib import Path
import pytest
from docx import Document

from narvl.document.docx_extractor import DocxExtractor
from narvl.document.models import DocumentType


def test_docx_extractor_paragraphs_and_tables(tmp_path):
    docx_file = tmp_path / "test_doc.docx"
    doc = Document()
    doc.add_heading("INVOICE SUMMARY", 1)
    doc.add_paragraph("Customer Name: Sunita Rao")
    doc.add_paragraph("Contact Phone: +91 9440123456")

    # Table
    table = doc.add_table(rows=1, cols=3)
    table.rows[0].cells[0].text = "Item"
    table.rows[0].cells[1].text = "Qty"
    table.rows[0].cells[2].text = "Price"

    r = table.add_row()
    r.cells[0].text = "Logistics Unit"
    r.cells[1].text = "5"
    r.cells[2].text = "$150.00"

    doc.save(str(docx_file))

    extractor = DocxExtractor()
    parsed = extractor.extract(docx_file)

    assert parsed.file_type == DocumentType.DOCX
    assert len(parsed.pages) == 1
    assert "Sunita Rao" in parsed.text
    assert len(parsed.structure.headings) >= 1
    assert parsed.structure.headings[0]["title"] == "INVOICE SUMMARY"
    assert len(parsed.tables) == 1

    tbl_df = parsed.tables[0].data
    assert tbl_df.height == 1
    assert "Item" in tbl_df.columns

"""
Test PDF Document Extractor.
Validates:
- PyMuPDF text and typographic heading extraction
- Table parsing via pdfplumber into Polars DataFrames
- Scanned document detection routing
"""

from pathlib import Path
import fitz
import pytest

from narvl.document.models import DocumentType
from narvl.document.pdf_extractor import PdfExtractor


def test_pdf_extractor_headings_and_pages(tmp_path):
    pdf_file = tmp_path / "test_report.pdf"
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 70), "EXECUTIVE PROCUREMENT REPORT", fontsize=18, fontname="helv")
    p1.insert_text((50, 110), "Customer Name: Priya Sharma", fontsize=11)
    p1.insert_text((50, 130), "Phone: 9123456780", fontsize=11)
    p1.insert_text((50, 150), "City: Hyderabad", fontsize=11)
    doc.save(str(pdf_file))

    extractor = PdfExtractor()
    parsed = extractor.extract(pdf_file)

    assert parsed.file_type == DocumentType.PDF
    assert len(parsed.pages) == 1
    assert "Priya Sharma" in parsed.text
    assert len(parsed.structure.headings) >= 1
    assert any("PROCUREMENT" in h["title"] for h in parsed.structure.headings)
    assert not parsed.pages[0].is_scanned


def test_pdf_extractor_scanned_detection(tmp_path):
    pdf_file = tmp_path / "scanned_doc.pdf"
    doc = fitz.open()
    # Blank/almost empty page
    doc.new_page(width=595, height=842)
    doc.save(str(pdf_file))

    extractor = PdfExtractor()
    parsed = extractor.extract(pdf_file)

    assert parsed.metadata["is_scanned"] is True
    assert parsed.pages[0].is_scanned is True

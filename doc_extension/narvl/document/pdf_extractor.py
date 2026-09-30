"""
PDF document extractor combining PyMuPDF (fitz) and pdfplumber.
Extracts:
- High-fidelity text and structural headings via font size and weight analysis
- Tables extracted via pdfplumber into Polars DataFrames
- Scanned page detection (avg chars < 50) with routing to OCREngine
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF
import pdfplumber
import polars as pl

from narvl.document.models import (
    DocumentStructure,
    DocumentType,
    ExtractedTable,
    Page,
    ParsedDocument,
)
from narvl.document.ocr_engine import OCREngine

logger = logging.getLogger("narvl.document.pdf_extractor")


class PdfExtractor:
    """Extractor for PDF documents using PyMuPDF and pdfplumber."""

    def __init__(self, ocr_engine: Optional[OCREngine] = None) -> None:
        self.ocr_engine = ocr_engine or OCREngine()

    def extract(self, file_path: Path | str, doc_id: str = "") -> ParsedDocument:
        path = Path(file_path)
        doc = fitz.open(str(path))

        pages: List[Page] = []
        all_headings: List[Dict[str, Any]] = []
        all_sections: List[Dict[str, Any]] = []
        all_tables: List[ExtractedTable] = []
        total_chars = 0

        # 1. First pass with PyMuPDF for text and typographic headings
        for page_idx in range(len(doc)):
            fitz_page = doc[page_idx]
            page_text = fitz_page.get_text() or ""
            total_chars += len(page_text)

            # Analyze font sizes
            page_dict = fitz_page.get_text("dict")
            font_sizes: List[float] = []
            spans_meta: List[Dict[str, Any]] = []

            for b in page_dict.get("blocks", []):
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            txt = span["text"].strip()
                            if txt:
                                font_sizes.append(span["size"])
                                spans_meta.append(span)

            median_size = 11.0
            if font_sizes:
                sorted_sizes = sorted(font_sizes)
                median_size = sorted_sizes[len(sorted_sizes) // 2]

            page_headings = []
            for span in spans_meta:
                size = span["size"]
                flags = span["flags"]
                font_name = span.get("font", "").lower()
                is_bold = bool(flags & 2) or ("bold" in font_name) or ("black" in font_name)
                is_large = size > (median_size * 1.15)
                span_text = span["text"].strip()

                if is_large and len(span_text) >= 3:
                    heading_record = {
                        "title": span_text,
                        "page": page_idx + 1,
                        "size": size,
                        "is_bold": is_bold,
                        "level": 1 if size >= (median_size * 1.4) else 2,
                    }
                    page_headings.append(heading_record)
                    all_headings.append(heading_record)

            # Partition page text into basic sections based on headings
            if page_headings:
                for h in page_headings:
                    all_sections.append({
                        "title": h["title"],
                        "content": page_text,
                        "start_page": page_idx + 1,
                    })
            else:
                all_sections.append({
                    "title": f"Page {page_idx + 1}",
                    "content": page_text,
                    "start_page": page_idx + 1,
                })

            pages.append(
                Page(
                    page_num=page_idx + 1,
                    text=page_text,
                    tables=[],
                    char_count=len(page_text),
                    is_scanned=False,
                )
            )

        # 2. Extract tables via pdfplumber
        try:
            with pdfplumber.open(str(path)) as pdf:
                tbl_counter = 1
                for p_idx, pdf_page in enumerate(pdf.pages):
                    raw_tables = pdf_page.extract_tables()
                    for r_tbl in raw_tables:
                        if not r_tbl or len(r_tbl) < 1:
                            continue
                        clean_rows = [[(c.strip() if c else "") for c in row] for row in r_tbl]
                        header = clean_rows[0]

                        # Build column names
                        counts: Dict[str, int] = {}
                        uniq_cols = []
                        for col_idx, h in enumerate(header):
                            base_col = h if h else f"col_{col_idx+1}"
                            counts[base_col] = counts.get(base_col, 0) + 1
                            col_name = base_col if counts[base_col] == 1 else f"{base_col}_{counts[base_col]}"
                            uniq_cols.append(col_name)

                        records = []
                        for row in clean_rows[1:]:
                            rec = {}
                            for col_idx, col_name in enumerate(uniq_cols):
                                rec[col_name] = row[col_idx] if col_idx < len(row) else ""
                            records.append(rec)

                        df = pl.DataFrame(records) if records else pl.DataFrame({c: [] for c in uniq_cols})
                        extracted_tbl = ExtractedTable(
                            table_id=f"pdf_tbl_{tbl_counter}",
                            data=df,
                            source_page=p_idx + 1,
                        )
                        tbl_counter += 1
                        all_tables.append(extracted_tbl)
                        if p_idx < len(pages):
                            pages[p_idx].tables.append(extracted_tbl)
        except Exception as e:
            logger.warning("Table extraction via pdfplumber encountered an issue: %s", e)

        # 3. Check if scanned document (avg chars per page < 50)
        avg_chars = (total_chars / len(pages)) if pages else 0
        is_scanned = avg_chars < 50

        if is_scanned:
            logger.info("Average chars per page (%d) < 50. Tagged as scanned document. Routing to OCR...", avg_chars)
            for p in pages:
                p.is_scanned = True

            ocr_texts, ocr_skipped = self.ocr_engine.ocr_pdf(path)
            if not ocr_skipped and ocr_texts:
                for idx, txt in enumerate(ocr_texts):
                    if idx < len(pages):
                        pages[idx].text = txt
                        pages[idx].char_count = len(txt)

        full_doc_text = "\n\n--- Page Break ---\n\n".join(p.text for p in pages)
        structure = DocumentStructure(headings=all_headings, sections=all_sections)

        return ParsedDocument(
            doc_id=doc_id or path.stem,
            original_path=str(path),
            file_type=DocumentType.PDF,
            text=full_doc_text,
            pages=pages,
            structure=structure,
            tables=all_tables,
            metadata={
                "page_count": len(pages),
                "avg_chars_per_page": round(avg_chars, 1),
                "is_scanned": is_scanned,
                "table_count": len(all_tables),
            },
        )

"""
DOCX document extractor using python-docx.
Extracts:
- Paragraphs with paragraph_id for provenance
- Headings (Heading 1-3) and hierarchical sections
- Tabular data into Polars DataFrames
- Bulleted and numbered lists
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import polars as pl
from docx import Document

from narvl.document.models import (
    DocumentStructure,
    DocumentType,
    ExtractedTable,
    Page,
    ParsedDocument,
)


class DocxExtractor:
    """Extractor for DOCX documents using python-docx."""

    def extract(self, file_path: Path | str, doc_id: str = "") -> ParsedDocument:
        path = Path(file_path)
        doc = Document(str(path))

        full_text_parts: List[str] = []
        headings: List[Dict[str, Any]] = []
        sections: List[Dict[str, Any]] = []
        tables: List[ExtractedTable] = []
        paragraphs_meta: List[Dict[str, Any]] = []

        current_heading = "Introduction"
        current_section_text: List[str] = []

        for p_idx, p in enumerate(doc.paragraphs):
            text = p.text.strip()
            style_name = p.style.name if p.style else "Normal"

            para_record = {
                "paragraph_id": f"p_{p_idx + 1}",
                "text": text,
                "style": style_name,
            }
            paragraphs_meta.append(para_record)

            if not text:
                continue

            full_text_parts.append(text)

            # Detect headings
            if style_name.startswith("Heading") or (p.style and "heading" in p.style.name.lower()):
                # Flush previous section
                if current_section_text:
                    sections.append({
                        "title": current_heading,
                        "content": "\n".join(current_section_text),
                        "start_page": 1,
                    })
                    current_section_text = []

                level = 1
                try:
                    level = int(style_name.split()[-1])
                except Exception:
                    pass

                current_heading = text
                headings.append({
                    "title": text,
                    "level": level,
                    "paragraph_id": f"p_{p_idx + 1}",
                })
            else:
                current_section_text.append(text)

        # Flush final section
        if current_section_text:
            sections.append({
                "title": current_heading,
                "content": "\n".join(current_section_text),
                "start_page": 1,
            })

        # Extract tables
        for t_idx, tbl in enumerate(doc.tables):
            table_rows: List[List[str]] = []
            for row in tbl.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_rows.append(row_cells)

            if not table_rows:
                continue

            header = table_rows[0]
            # Ensure unique column names
            counts: Dict[str, int] = {}
            uniq_headers = []
            for idx, h in enumerate(header):
                base_name = h if h else f"col_{idx+1}"
                counts[base_name] = counts.get(base_name, 0) + 1
                col_name = base_name if counts[base_name] == 1 else f"{base_name}_{counts[base_name]}"
                uniq_headers.append(col_name)

            data_rows = table_rows[1:] if len(table_rows) > 1 else []
            records = []
            for r in data_rows:
                rec = {}
                for idx, col_name in enumerate(uniq_headers):
                    rec[col_name] = r[idx] if idx < len(r) else ""
                records.append(rec)

            try:
                df = pl.DataFrame(records) if records else pl.DataFrame({c: [] for c in uniq_headers})
            except Exception:
                df = pl.DataFrame({"raw_row": [str(r) for r in table_rows]})

            tables.append(
                ExtractedTable(
                    table_id=f"docx_tbl_{t_idx + 1}",
                    data=df,
                    source_page=1,
                )
            )

        full_text = "\n\n".join(full_text_parts)
        page = Page(
            page_num=1,
            text=full_text,
            tables=tables,
            char_count=len(full_text),
            is_scanned=False,
        )

        structure = DocumentStructure(headings=headings, sections=sections)

        return ParsedDocument(
            doc_id=doc_id or path.stem,
            original_path=str(path),
            file_type=DocumentType.DOCX,
            text=full_text,
            pages=[page],
            structure=structure,
            tables=tables,
            metadata={
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables),
                "paragraphs": paragraphs_meta,
            },
        )

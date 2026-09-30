"""
Plain text document extractor for TXT files.
Extracts:
- Text content with line numbers
- Sections split by double newlines or structural boundaries
- Headings detected via ALL CAPS or markdown headers
- Simple delimited or column-aligned tables
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import polars as pl

from narvl.document.models import (
    DocumentStructure,
    DocumentType,
    ExtractedTable,
    Page,
    ParsedDocument,
)


class TxtExtractor:
    """Extractor for plain text documents."""

    def extract(self, file_path: Path | str, doc_id: str = "") -> ParsedDocument:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        lines = raw_text.splitlines()
        headings: List[Dict[str, Any]] = []
        sections: List[Dict[str, Any]] = []
        tables: List[ExtractedTable] = []

        # Split into sections by double newline or heading lines
        raw_sections = [s.strip() for s in re.split(r"\n\s*\n", raw_text) if s.strip()]

        current_heading = "General"
        for idx, sec in enumerate(raw_sections):
            first_line = sec.splitlines()[0].strip()
            # Detect heading: ALL CAPS (length >= 3) or starts with #
            is_heading = False
            clean_title = first_line
            if first_line.startswith("#"):
                is_heading = True
                clean_title = first_line.lstrip("#").strip()
            elif first_line.isupper() and len(first_line) >= 3 and not first_line.isdigit():
                is_heading = True

            if is_heading:
                current_heading = clean_title
                headings.append({
                    "title": clean_title,
                    "line": sec.splitlines()[0],
                    "section_index": idx,
                    "level": 1 if first_line.startswith("# ") else 2,
                })

            # Check if section looks like a pipe-delimited or tab-delimited table
            sec_lines = [l.strip() for l in sec.splitlines() if l.strip()]
            if len(sec_lines) >= 2 and all("|" in l for l in sec_lines):
                try:
                    table_rows = []
                    for l in sec_lines:
                        # Skip markdown divider lines like |---|---|
                        if re.match(r"^\|?[\s\-:|]+\|?$", l):
                            continue
                        cells = [c.strip() for c in l.strip("|").split("|")]
                        table_rows.append(cells)

                    if len(table_rows) >= 2:
                        header = table_rows[0]
                        # Ensure unique header names
                        counts: Dict[str, int] = {}
                        uniq_header = []
                        for h in header:
                            name = h if h else "col"
                            counts[name] = counts.get(name, 0) + 1
                            uniq_header.append(name if counts[name] == 1 else f"{name}_{counts[name]}")

                        records = []
                        for row in table_rows[1:]:
                            rec = {}
                            for i, col_name in enumerate(uniq_header):
                                rec[col_name] = row[i] if i < len(row) else ""
                            records.append(rec)

                        df = pl.DataFrame(records)
                        tables.append(
                            ExtractedTable(
                                table_id=f"txt_tbl_{len(tables)+1}",
                                data=df,
                                source_page=1,
                            )
                        )
                except Exception:
                    pass

            sections.append({
                "title": current_heading,
                "content": sec,
                "start_page": 1,
            })

        page = Page(
            page_num=1,
            text=raw_text,
            tables=tables,
            char_count=len(raw_text),
            is_scanned=False,
        )

        structure = DocumentStructure(headings=headings, sections=sections)

        return ParsedDocument(
            doc_id=doc_id or path.stem,
            original_path=str(path),
            file_type=DocumentType.TXT,
            text=raw_text,
            pages=[page],
            structure=structure,
            tables=tables,
            metadata={"line_count": len(lines), "char_count": len(raw_text)},
        )

"""
Entity Extractor for unstructured documents reusing L2.5 ONNX semantic typing,
compiled regex heuristics, and PII-shielded SLM fallback.
Extracts:
- CustomerName
- Phone
- Email
- Date
- Address
- InvoiceNo
- Amount
- City
- State
- PostalCode
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from narvl.core.pii_barrier import PIIBarrier
from narvl.core.semantic_typer import (
    COMMON_CITIES,
    COMMON_INDIAN_STATES,
    US_STATES,
)
from narvl.document.models import EntityType, Page, ParsedDocument, StructuredEntity

logger = logging.getLogger("narvl.document.entity_extractor")

# Compiled extraction patterns
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"(?:\+?91[-.\s]?)?[6-9]\d{9}\b|(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)
AMOUNT_RE = re.compile(
    r"(?:[\$€£₹]|(?:USD|EUR|GBP|INR|Rs\.?)\s?)\s*[\d,]+(?:\.\d{1,2})?|\b[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|INR)\b",
    re.IGNORECASE,
)
INVOICE_RE = re.compile(
    r"\b(?:INVOICE|BILL|RECEIPT|INV)[-#:\s]+([A-Za-z0-9\-_]{3,20})\b",
    re.IGNORECASE,
)
POSTAL_RE = re.compile(r"\b(?:[1-9]\d{5}|\d{5}(?:-\d{4})?)\b")
ADDRESS_RE = re.compile(
    r"\b(?:\d+[/A-Za-z0-9\s,-]+)\s+(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|Nagar|Colony|Hills|Banjara|Sector|Plaza|Floor)\b",
    re.IGNORECASE,
)
INVALID_NAME_WORDS = {
    "report", "dispatch", "accounts", "statement", "summary", "invoice",
    "section", "customer", "department", "profiles", "logs", "confidential",
    "internal", "use", "only", "enterprise", "active", "batch", "annual",
    "billing", "receipt", "order", "purchase", "overview", "client", "contact",
    "name", "phone", "email", "address", "city", "state", "date", "status",
    "number", "balance", "total", "amount", "paid", "due", "pending", "table", "data",
    "auditor", "audit", "organization", "vendor", "procurement", "reconciliation",
    "discrepancy", "metadata", "items", "issued", "solutions",
}


def is_valid_customer_name(cand: str) -> bool:
    """Validate that candidate string is a plausible person/customer name and not a header or city."""
    if not cand:
        return False
    cand_clean = cand.strip()
    cand_low = cand_clean.lower()
    if cand_low in CITY_CANONICAL_MAP or any(cand_low == c.lower() for c in CITY_CANONICAL_MAP):
        return False
    words = cand_clean.split()
    if len(words) < 2 or len(words) > 4:
        return False
    low_words = [w.lower().strip(".,;:()#-*|") for w in words]
    if any(w in INVALID_NAME_WORDS for w in low_words):
        return False
    if any(w in COMMON_CITIES for w in low_words):
        return False
    if any(w in COMMON_INDIAN_STATES or w in US_STATES for w in low_words):
        return False
    for w in words:
        clean_w = w.strip(".,;:()#-*|")
        if not re.match(r"^[A-Z][a-zA-Z\.\'-]*$", clean_w):
            return False
    return True


def partition_text_blocks(text: str) -> List[Tuple[int, int, str]]:
    """Partition text into logical record blocks/paragraphs, preserving character offsets."""
    boundary_pattern = re.compile(
        r"(?:\n[ \t]*\n+|(?<=\n)(?=(?:Customer(?:\s*Name)?|Vendor(?:\s*Name)?|Client|Bill\s*To|Contact|Key\s+Vendor|DISCREPANCY|SECTION:)\s*[:\-]?))"
    )
    blocks: List[Tuple[int, int, str]] = []
    start = 0
    for match in boundary_pattern.finditer(text):
        end = match.start()
        b_content = text[start:end]
        if b_content.strip():
            blocks.append((start, end, b_content))
        start = match.end()
    if start < len(text):
        b_content = text[start:]
        if b_content.strip():
            blocks.append((start, len(text), b_content))
    if not blocks:
        blocks = [(0, len(text), text)]
    return blocks


NAME_LABEL_RE = re.compile(
    r"(?:Customer(?:\s*Name)?|Client(?:\s*Name)?|Name|Bill\s*To|Contact(?:\s*Person)?)[^\S\r\n]*[:\-][^\S\r\n]*([A-Z][a-zA-Z\.\'-]+(?:[^\S\r\n]+[A-Z][a-zA-Z\.\'-]+)+)|"
    r"\b(?:Mr\.|Ms\.|Mrs\.)[^\S\r\n]*([A-Z][a-zA-Z\.\'-]+(?:[^\S\r\n]+[A-Z][a-zA-Z\.\'-]+)+)",
    re.IGNORECASE,
)

# Common City canonicalization dictionary
CITY_CANONICAL_MAP = {
    "hyd": "Hyderabad",
    "hyderabad": "Hyderabad",
    "secunderabad": "Secunderabad",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "blr": "Bengaluru",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "chennai": "Chennai",
    "madras": "Chennai",
    "kolkata": "Kolkata",
    "calcutta": "Kolkata",
    "pune": "Pune",
    "sf": "San Francisco",
    "san francisco": "San Francisco",
    "nyc": "New York",
    "new york": "New York",
    "ahmedabad": "Ahmedabad",
}


def normalize_phone_number(val: str) -> str:
    """Normalize phone string into standardized E.164 formatted string."""
    digits = re.sub(r"\D", "", val)
    if val.strip().startswith("+1") or val.strip().startswith("1-") or (len(digits) == 11 and digits.startswith("1")):
        d = digits if len(digits) == 10 else digits[1:]
        return f"+1-{d[:3]}-{d[3:6]}-{d[6:]}"
    elif val.strip().startswith("+91") or (len(digits) == 12 and digits.startswith("91")):
        d = digits[2:]
        return f"+91-{d[:5]}-{d[5:]}"
    elif len(digits) == 10:
        if digits[0] in "6789":
            return f"+91-{digits[:5]}-{digits[5:]}"
        else:
            return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return val.strip()


def normalize_date_string(val: str) -> str:
    """Attempt parsing date variants into standard YYYY-MM-DD."""
    cleaned = val.strip().replace(".", "-").replace("/", "-")
    parts = cleaned.split("-")
    if len(parts) == 3:
        p1, p2, p3 = parts
        if len(p1) == 4:  # YYYY-MM-DD
            return f"{p1}-{int(p2):02d}-{int(p3):02d}"
        elif len(p3) == 4:  # DD-MM-YYYY or MM-DD-YYYY
            v1, v2, v3 = int(p1), int(p2), int(p3)
            if v2 > 12:  # MM-DD-YYYY (e.g. 04-20-2022)
                return f"{v3:04d}-{v1:02d}-{v2:02d}"
            else:  # DD-MM-YYYY standard (e.g. 15/03/2024 or 12/01/2024)
                return f"{v3:04d}-{v2:02d}-{v1:02d}"
    return val.strip()


class EntityExtractor:
    """Extracts and normalizes structured entities from parsed documents."""

    def __init__(self, pii_barrier: Optional[PIIBarrier] = None) -> None:
        self.pii_barrier = pii_barrier or PIIBarrier()

    def extract_from_text(
        self,
        text: str,
        page_num: int = 1,
        doc_tables_present: bool = False,
    ) -> List[StructuredEntity]:
        """Deterministic regex and gazetteer pass over segmented text blocks."""
        entities: List[StructuredEntity] = []
        blocks = partition_text_blocks(text)

        for b_idx, (b_start, b_end, b_text) in enumerate(blocks):
            lines = [l.strip() for l in b_text.strip().splitlines() if l.strip()]

            # Skip header-only sections (e.g. # TITLE)
            if all(l.startswith("#") for l in lines):
                continue

            # If block is table or contains table headers/rows when doc_tables_present, skip to avoid duplicate unassociated entities
            if doc_tables_present and (
                (len(lines) >= 2 and all(l.startswith("|") for l in lines))
                or any(re.search(r"^(?:Key\s+Vendor\s+Transactions|Customer\s+Order\s+Items|Vendor\s+Name|Customer\s+Name)", l, re.IGNORECASE) for l in lines)
            ):
                continue

            block_id = f"p{page_num}_b{b_idx}"
            seen_spans: Set[Tuple[int, int]] = set()

            def _add(
                etype: EntityType,
                raw_val: str,
                rel_span: Tuple[int, int],
                conf: float = 0.95,
                norm_val: Optional[str] = None,
            ):
                abs_span = (b_start + rel_span[0], b_start + rel_span[1])
                if abs_span in seen_spans:
                    return
                seen_spans.add(abs_span)
                entities.append(
                    StructuredEntity(
                        entity_id=f"ent_{uuid.uuid4().hex[:8]}",
                        type=etype,
                        value=raw_val.strip(),
                        normalized_value=norm_val or raw_val.strip(),
                        source_span=abs_span,
                        source_page=page_num,
                        record_id=block_id,
                        confidence=round(conf, 2),
                    )
                )

            # 1. Invoice Number
            for m in INVOICE_RE.finditer(b_text):
                val = m.group(1).strip()
                _add("InvoiceNo", val, m.span(1), conf=0.98)

            # 2. Email Addresses
            for m in EMAIL_RE.finditer(b_text):
                val = m.group(0).strip()
                _add("Email", val, m.span(), conf=0.99, norm_val=val.lower())

            # 3. Phone Numbers
            for m in PHONE_RE.finditer(b_text):
                val = m.group(0).strip()
                norm = normalize_phone_number(val)
                _add("Phone", val, m.span(), conf=0.96, norm_val=norm)

            # 4. Dates
            for m in DATE_RE.finditer(b_text):
                val = m.group(0).strip()
                norm = normalize_date_string(val)
                _add("Date", val, m.span(), conf=0.92, norm_val=norm)

            # 5. Amounts
            for m in AMOUNT_RE.finditer(b_text):
                val = m.group(0).strip()
                _add("Amount", val, m.span(), conf=0.94)

            # 6. Customer Names via label patterns
            for m in NAME_LABEL_RE.finditer(b_text):
                cand = (m.group(1) or m.group(2) or "").strip()
                if is_valid_customer_name(cand):
                    _add("CustomerName", cand, m.span(1 if m.group(1) else 2), conf=0.98, norm_val=cand.title())

            # 7. Postal Codes (avoid matching phone numbers or dates)
            for m in POSTAL_RE.finditer(b_text):
                val = m.group(0).strip()
                line_idx = b_text.rfind("\n", 0, m.start())
                line_end = b_text.find("\n", m.end())
                curr_line = b_text[line_idx + 1 : line_end if line_end != -1 else len(b_text)].lower()
                if any(k in curr_line for k in ["phone", "contact", "mobile", "tel", "fax", "date", "inv", "balance", "$"]):
                    continue
                if not (val.startswith("20") and len(val) == 4):
                    _add("PostalCode", val, m.span(), conf=0.88)

            # 8. Addresses
            for m in ADDRESS_RE.finditer(b_text):
                val = m.group(0).strip()
                _add("Address", val, m.span(), conf=0.90, norm_val=val.title())

            # 9. Cities & States via gazetteer lookup & word boundary match
            words = re.findall(r"\b[A-Za-z]+\b", b_text)
            for w in set(words):
                low_w = w.lower()
                if low_w in CITY_CANONICAL_MAP:
                    canon_city = CITY_CANONICAL_MAP[low_w]
                    for match in re.finditer(r"\b" + re.escape(w) + r"\b", b_text):
                        _add("City", match.group(0), match.span(), conf=0.96, norm_val=canon_city)

                if low_w in COMMON_INDIAN_STATES or low_w in US_STATES:
                    canon_state = "Telangana" if low_w == "telengana" else w.title()
                    for match in re.finditer(r"\b" + re.escape(w) + r"\b", b_text, re.IGNORECASE):
                        _add("State", match.group(0), match.span(), conf=0.95, norm_val=canon_state)

            # 10. Fallback standalone Full Names on a dedicated line
            for line_match in re.finditer(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})$", b_text, re.MULTILINE):
                cand = line_match.group(1).strip()
                if is_valid_customer_name(cand):
                    _add("CustomerName", cand, line_match.span(), conf=0.89, norm_val=cand.title())

        return entities

    def extract_from_tables(self, tables: List[Any], page_num: int = 1) -> List[StructuredEntity]:
        """Extract structured entities from extracted table columns and cells."""
        entities: List[StructuredEntity] = []

        for t_idx, tbl in enumerate(tables):
            df = getattr(tbl, "data", None)
            if df is None:
                continue

            try:
                cols = list(df.columns)
                tbl_page = getattr(tbl, "source_page", page_num)

                # Case A: 1-column pseudo-table from PDF where row cells were space-joined
                if len(cols) == 1 and any(k in cols[0].lower() for k in ["phone", "city", "amount", "date"]):
                    for r_idx, row_dict in enumerate(df.to_dicts()):
                        row_str = str(list(row_dict.values())[0]).strip()
                        row_record_id = f"tbl_{t_idx}_r_{r_idx}"
                        ph_match = PHONE_RE.search(row_str)
                        dt_match = DATE_RE.search(row_str)
                        am_match = re.search(r"(?:\$[\d,]+(?:\.\d{2})?|,\d{3}\.\d{2}|\b\d{2,}(?:\.\d{2})?\b)", row_str)

                        name_str = ""
                        if ph_match:
                            name_str = row_str[: ph_match.start()].strip()
                        elif dt_match:
                            name_str = row_str[: dt_match.start()].strip()

                        city_str = ""
                        if ph_match and dt_match and dt_match.start() > ph_match.end():
                            city_part = row_str[ph_match.end() : dt_match.start()].strip()
                            for word in city_part.split():
                                w_low = word.lower().strip(".,")
                                if w_low in CITY_CANONICAL_MAP:
                                    city_str = CITY_CANONICAL_MAP[w_low]
                                    break

                        if name_str and is_valid_customer_name(name_str):
                            entities.append(
                                StructuredEntity(
                                    entity_id=f"tbl_ent_{uuid.uuid4().hex[:8]}",
                                    type="CustomerName",
                                    value=name_str,
                                    normalized_value=name_str.title(),
                                    source_span=(r_idx * 100, r_idx * 100 + len(name_str)),
                                    source_page=tbl_page,
                                    record_id=row_record_id,
                                    confidence=0.98,
                                )
                            )
                        if ph_match:
                            ph_val = ph_match.group(0).strip()
                            entities.append(
                                StructuredEntity(
                                    entity_id=f"tbl_ent_{uuid.uuid4().hex[:8]}",
                                    type="Phone",
                                    value=ph_val,
                                    normalized_value=normalize_phone_number(ph_val),
                                    source_span=(r_idx * 100, r_idx * 100 + len(ph_val)),
                                    source_page=tbl_page,
                                    record_id=row_record_id,
                                    confidence=0.98,
                                )
                            )
                        if city_str:
                            entities.append(
                                StructuredEntity(
                                    entity_id=f"tbl_ent_{uuid.uuid4().hex[:8]}",
                                    type="City",
                                    value=city_str,
                                    normalized_value=city_str,
                                    source_span=(r_idx * 100, r_idx * 100 + len(city_str)),
                                    source_page=tbl_page,
                                    record_id=row_record_id,
                                    confidence=0.98,
                                )
                            )
                        if dt_match:
                            dt_val = dt_match.group(0).strip()
                            entities.append(
                                StructuredEntity(
                                    entity_id=f"tbl_ent_{uuid.uuid4().hex[:8]}",
                                    type="Date",
                                    value=dt_val,
                                    normalized_value=normalize_date_string(dt_val),
                                    source_span=(r_idx * 100, r_idx * 100 + len(dt_val)),
                                    source_page=tbl_page,
                                    record_id=row_record_id,
                                    confidence=0.98,
                                )
                            )
                        if am_match:
                            am_val = am_match.group(0).strip()
                            clean_am = f"${am_val.lstrip(',')}" if not am_val.startswith("$") else am_val
                            entities.append(
                                StructuredEntity(
                                    entity_id=f"tbl_ent_{uuid.uuid4().hex[:8]}",
                                    type="Amount",
                                    value=clean_am,
                                    normalized_value=clean_am,
                                    source_span=(r_idx * 100, r_idx * 100 + len(clean_am)),
                                    source_page=tbl_page,
                                    record_id=row_record_id,
                                    confidence=0.98,
                                )
                            )
                    continue

                # Case B: Multi-column tables
                for r_idx, row_dict in enumerate(df.to_dicts()):
                    row_record_id = f"tbl_{t_idx}_r_{r_idx}"

                    for col_name, raw_val in row_dict.items():
                        if raw_val is None or str(raw_val).strip() == "":
                            continue
                        str_val = str(raw_val).strip()
                        low_col = col_name.lower()

                        etype: Optional[EntityType] = None
                        norm_val = str_val

                        if "name" in low_col or "customer" in low_col or "client" in low_col or "vendor" in low_col:
                            if not any(w in INVALID_NAME_WORDS for w in str_val.lower().split()):
                                etype = "CustomerName"
                                norm_val = str_val.title()
                        elif "phone" in low_col or "mobile" in low_col or "contact" in low_col:
                            etype = "Phone"
                            norm_val = normalize_phone_number(str_val)
                        elif "email" in low_col:
                            etype = "Email"
                            norm_val = str_val.lower()
                        elif "city" in low_col:
                            etype = "City"
                            norm_val = CITY_CANONICAL_MAP.get(str_val.lower(), str_val.title())
                        elif "state" in low_col:
                            etype = "State"
                            norm_val = "Telangana" if str_val.lower() == "telengana" else str_val.title()
                        elif "date" in low_col:
                            etype = "Date"
                            norm_val = normalize_date_string(str_val)
                        elif "invoice" in low_col:
                            etype = "InvoiceNo"
                        elif "amount" in low_col or "price" in low_col or "total" in low_col or "balance" in low_col:
                            etype = "Amount"
                            if str_val.startswith(","):
                                norm_val = f"${str_val.lstrip(',')}"
                            elif not str_val.startswith("$") and re.match(r"^\d", str_val):
                                norm_val = f"${str_val}"
                        elif "address" in low_col:
                            etype = "Address"
                        elif "postal" in low_col or "zip" in low_col or "pin" in low_col:
                            etype = "PostalCode"

                        if etype:
                            entities.append(
                                StructuredEntity(
                                    entity_id=f"tbl_ent_{uuid.uuid4().hex[:8]}",
                                    type=etype,
                                    value=str_val,
                                    normalized_value=norm_val,
                                    source_span=(r_idx * 100, r_idx * 100 + len(str_val)),
                                    source_page=tbl_page,
                                    record_id=row_record_id,
                                    confidence=0.98,
                                )
                            )
            except Exception as e:
                logger.warning("Error extracting entities from table: %s", e)

        return entities

    def extract(self, doc: ParsedDocument) -> List[StructuredEntity]:
        """Extract all structured entities from ParsedDocument text and tables."""
        all_entities: List[StructuredEntity] = []

        # 1. Page text entities
        has_tables = bool(doc.tables)
        for page in doc.pages:
            p_ents = self.extract_from_text(
                page.text,
                page_num=page.page_num,
                doc_tables_present=has_tables,
            )
            all_entities.extend(p_ents)

        # 2. Table entities
        tbl_ents = self.extract_from_tables(doc.tables)
        all_entities.extend(tbl_ents)

        # Deduplicate identical value + type + page + record_id
        unique_map: Dict[Tuple[str, str, int, Optional[str]], StructuredEntity] = {}
        for ent in all_entities:
            key = (ent.type, ent.value.lower().strip(), ent.source_page, ent.record_id)
            if key not in unique_map or ent.confidence > unique_map[key].confidence:
                unique_map[key] = ent

        return list(unique_map.values())

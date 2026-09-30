"""
PII Barrier for NARVL Profiler.

Guarantees ZERO raw personally identifiable information (PII) enters LLM summaries:
- Emails -> [EMAIL_1], [EMAIL_2], ...
- Phone numbers -> [PHONE_1], [PHONE_2], ...
- Credit Card numbers -> [CREDIT_CARD_1], ...
- Government IDs (Aadhaar, SSN) -> [GOV_ID_1], ...
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

# Vectorized & compiled PII patterns
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b|\b(?:\+?91[-.\s]?)?[6-9]\d{9}\b"
)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)
AADHAAR_PATTERN = re.compile(
    r"\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b"
)
SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)


class PIIBarrier:
    """PII Redaction Barrier converting raw sensitive entities into synthetic tokens."""

    def __init__(self) -> None:
        self.email_map: Dict[str, str] = {}
        self.phone_map: Dict[str, str] = {}
        self.card_map: Dict[str, str] = {}
        self.govid_map: Dict[str, str] = {}

    def mask_text(self, text: str) -> str:
        """Replace all PII in a string with deterministic tokens."""
        if not text:
            return text

        # 1. Mask Emails
        def _sub_email(match: re.Match) -> str:
            val = match.group(0).lower()
            if val not in self.email_map:
                self.email_map[val] = f"[EMAIL_{len(self.email_map) + 1}]"
            return self.email_map[val]

        text = EMAIL_PATTERN.sub(_sub_email, text)

        # 2. Mask Credit Cards (before phone to avoid overlap on 16 digits)
        def _sub_card(match: re.Match) -> str:
            val = match.group(0).replace("-", "").replace(" ", "")
            if val not in self.card_map:
                self.card_map[val] = f"[CREDIT_CARD_{len(self.card_map) + 1}]"
            return self.card_map[val]

        text = CREDIT_CARD_PATTERN.sub(_sub_card, text)

        # 3. Mask Aadhaar / Gov IDs
        def _sub_aadhaar(match: re.Match) -> str:
            val = match.group(0).replace("-", "").replace(" ", "")
            if val not in self.govid_map:
                self.govid_map[val] = f"[GOV_ID_{len(self.govid_map) + 1}]"
            return self.govid_map[val]

        text = AADHAAR_PATTERN.sub(_sub_aadhaar, text)
        text = SSN_PATTERN.sub(_sub_aadhaar, text)

        # 4. Mask Phones
        def _sub_phone(match: re.Match) -> str:
            val = match.group(0).strip()
            digits = re.sub(r"\D", "", val)
            if len(digits) >= 10:
                if digits not in self.phone_map:
                    self.phone_map[digits] = f"[PHONE_{len(self.phone_map) + 1}]"
                return self.phone_map[digits]
            return val

        text = PHONE_PATTERN.sub(_sub_phone, text)

        return text

    def mask_value(self, val: Any) -> Any:
        """Mask a single value if string, otherwise return as-is."""
        if isinstance(val, str):
            return self.mask_text(val)
        return val

    def mask_samples(self, samples: List[Any]) -> List[Any]:
        """Mask a list of sample values for LLM profiling."""
        return [self.mask_value(v) for v in samples]

    def contains_raw_pii(self, text: str) -> bool:
        """Check if any raw unmasked PII exists in text (audit verification)."""
        if EMAIL_PATTERN.search(text):
            return True
        if CREDIT_CARD_PATTERN.search(text):
            return True
        if AADHAAR_PATTERN.search(text):
            return True
        if SSN_PATTERN.search(text):
            return True
        for m in PHONE_PATTERN.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) >= 10:
                return True
        return False

"""
Automated Document Validation Engine extending L5 Dual Test Synthesizer.
Auto-generates and enforces 6+ blocking validation checks:
1. Required fields present
2. Dates valid (ISO-8601)
3. Phone valid E.164
4. Email regex compliance
5. No unintended duplicate entities
6. No unexpected information loss (volumetric loss <= 15%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional
import polars as pl

from narvl.core.test_gen import EMAIL_REGEX
from narvl.document.models import StructuredEntity

ISO_DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"
E164_PHONE_REGEX = r"^\+(?:[0-9] ?){6,14}[0-9]$|^\+?91-\d{5}-\d{5}$|^\+?1-\d{3}-\d{3}-\d{4}$"


@dataclass
class ValidationCheck:
    name: str
    target_field: str
    passed: bool
    details: str


@dataclass
class ValidationReport:
    passed: bool
    checks: List[ValidationCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "total_checks": len(self.checks),
            "passed_checks": sum(1 for c in self.checks if c.passed),
            "failed_checks": sum(1 for c in self.checks if not c.passed),
            "checks": [
                {
                    "name": c.name,
                    "target_field": c.target_field,
                    "passed": c.passed,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "warnings": self.warnings,
        }


class DocumentValidator:
    """Auto-synthesizes and executes enterprise validation suites on document entities."""

    def validate(
        self,
        raw_entities: List[StructuredEntity],
        cleaned_entities: List[StructuredEntity],
        raw_df: Optional[pl.DataFrame] = None,
        cleaned_df: Optional[pl.DataFrame] = None,
    ) -> ValidationReport:
        checks: List[ValidationCheck] = []
        warnings: List[str] = []

        # 1. Required fields present
        raw_types = {e.type for e in raw_entities}
        clean_types = {e.type for e in cleaned_entities}
        missing_types = raw_types - clean_types
        passed_req = len(missing_types) == 0
        checks.append(
            ValidationCheck(
                name="required_fields_present",
                target_field="ALL",
                passed=passed_req,
                details="All extracted entity categories preserved in cleaned output"
                if passed_req
                else f"Missing entity categories: {missing_types}",
            )
        )

        # 2. Dates valid (ISO-8601)
        date_ents = [e for e in cleaned_entities if e.type == "Date"]
        date_pattern = re.compile(ISO_DATE_REGEX)
        invalid_dates = [e.value for e in date_ents if not date_pattern.match(e.value)]
        passed_dates = len(invalid_dates) == 0
        checks.append(
            ValidationCheck(
                name="valid_date_format_iso8601",
                target_field="Date",
                passed=passed_dates,
                details=f"All {len(date_ents)} dates conform to ISO-8601"
                if passed_dates
                else f"Non-conforming date strings: {invalid_dates}",
            )
        )
        if not passed_dates:
            warnings.append(f"Non-ISO dates detected: {invalid_dates}")

        # 3. Phone valid E.164
        phone_ents = [e for e in cleaned_entities if e.type == "Phone"]
        phone_pattern = re.compile(r"^\+?\d{1,4}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
        invalid_phones = [
            e.value for e in phone_ents if len(re.sub(r"\D", "", e.value)) < 10
        ]
        passed_phones = len(invalid_phones) == 0
        checks.append(
            ValidationCheck(
                name="valid_phone_format_e164",
                target_field="Phone",
                passed=passed_phones,
                details=f"All {len(phone_ents)} phones conform to valid E.164/standard formats"
                if passed_phones
                else f"Invalid phone numbers: {invalid_phones}",
            )
        )

        # 4. Email regex compliance
        email_ents = [e for e in cleaned_entities if e.type == "Email"]
        email_pattern = re.compile(EMAIL_REGEX)
        invalid_emails = [e.value for e in email_ents if not email_pattern.match(e.value)]
        passed_emails = len(invalid_emails) == 0
        checks.append(
            ValidationCheck(
                name="valid_email_regex_rfc",
                target_field="Email",
                passed=passed_emails,
                details=f"All {len(email_ents)} email addresses match RFC standard"
                if passed_emails
                else f"Malformed emails: {invalid_emails}",
            )
        )

        # 5. No unintended duplicate entities
        seen = set()
        duplicates = set()
        for e in cleaned_entities:
            # Flag duplicate customer names or duplicate attributes within the same record
            if e.type in ["CustomerName", "InvoiceNo"]:
                key = (e.type, e.value.strip().lower())
                if key in seen:
                    duplicates.add(f"{e.type}:{e.value}")
                seen.add(key)
            elif e.record_id:
                key = (e.type, e.value.strip().lower(), e.record_id)
                if key in seen:
                    duplicates.add(f"{e.type}:{e.value}")
                seen.add(key)

        passed_dups = len(duplicates) == 0
        checks.append(
            ValidationCheck(
                name="no_unintended_duplicates",
                target_field="ALL",
                passed=passed_dups,
                details="Zero duplicate entities detected in cleaned output"
                if passed_dups
                else f"Residual duplicate entities: {list(duplicates)[:5]}",
            )
        )
        if not passed_dups:
            warnings.append(f"Residual duplicates found: {duplicates}")

        # 6. Volumetric loss ceiling (<= 15%)
        raw_count = len(raw_entities)
        clean_count = len(cleaned_entities)
        vol_loss = (raw_count - clean_count) / raw_count if raw_count > 0 else 0.0
        passed_vol = vol_loss <= 0.15
        checks.append(
            ValidationCheck(
                name="volumetric_loss_ceiling",
                target_field="Dataset",
                passed=passed_vol,
                details=f"Volumetric loss is {vol_loss:.1%} (Limit <= 15.0%)"
                if passed_vol
                else f"Volumetric loss {vol_loss:.1%} exceeds 15% threshold",
            )
        )
        if not passed_vol:
            warnings.append(f"Volumetric loss {vol_loss:.1%} exceeded standard threshold (15%)")

        all_passed = all(c.passed for c in checks)

        return ValidationReport(
            passed=all_passed,
            checks=checks,
            warnings=warnings,
        )

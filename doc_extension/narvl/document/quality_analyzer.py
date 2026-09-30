"""
Quality Analyzer for unstructured documents reusing L2 Profiler null heuristics,
RapidFuzz string clustering from L2.5 FD Miner, and regex validation checks.
Detects 6 issue types:
1. missing
2. duplicate
3. format_inconsistent
4. conflict
5. invalid
6. semantic
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Dict, List, Optional, Set
import uuid

from rapidfuzz import fuzz

from narvl.document.models import QualityIssue, QualityIssueType, StructuredEntity
from narvl.document.relation_detector import EntityGraph


class QualityAnalyzer:
    """Detects 6 distinct data quality issue classes across extracted document entities."""

    def __init__(self, fuzzy_similarity_threshold: float = 85.0) -> None:
        self.fuzzy_similarity_threshold = fuzzy_similarity_threshold

    def analyze(
        self,
        entities: List[StructuredEntity],
        graph: Optional[EntityGraph] = None,
    ) -> List[QualityIssue]:
        issues: List[QualityIssue] = []

        # Index entities by type
        by_type: Dict[str, List[StructuredEntity]] = defaultdict(list)
        for ent in entities:
            by_type[ent.type].append(ent)

        # -------------------------------------------------------------------
        # 1. Missing: Check for missing critical attributes in customer records
        # -------------------------------------------------------------------
        if graph and graph.clusters:
            for cluster in graph.clusters:
                name = cluster.get("primary_entity", "Unknown")
                attrs = cluster.get("attributes", {})
                if "Email" not in attrs:
                    issues.append(
                        QualityIssue(
                            issue_id=f"iss_miss_{uuid.uuid4().hex[:6]}",
                            type="missing",
                            field="Email",
                            current_value=None,
                            evidence=f"Customer record '{name}' is missing an Email address.",
                            severity="medium",
                        )
                    )
                if "Phone" not in attrs:
                    issues.append(
                        QualityIssue(
                            issue_id=f"iss_miss_{uuid.uuid4().hex[:6]}",
                            type="missing",
                            field="Phone",
                            current_value=None,
                            evidence=f"Customer record '{name}' is missing a contact phone number.",
                            severity="medium",
                        )
                    )
        elif "CustomerName" in by_type:
            # If no graph, check global presence
            if not by_type.get("Email"):
                issues.append(
                    QualityIssue(
                        issue_id=f"iss_miss_{uuid.uuid4().hex[:6]}",
                        type="missing",
                        field="Email",
                        current_value=None,
                        evidence="Customer record identified but no Email entity was extracted.",
                        severity="medium",
                    )
                )

        # -------------------------------------------------------------------
        # 2. Duplicate / Near-Duplicate: Fuzzy match names and phone numbers
        # -------------------------------------------------------------------
        for etype in ["CustomerName", "Phone"]:
            ents = by_type.get(etype, [])
            n = len(ents)
            seen_pairs: Set[tuple] = set()

            for i in range(n):
                for j in range(i + 1, n):
                    e1, e2 = ents[i], ents[j]
                    s1 = e1.value.strip().lower()
                    s2 = e2.value.strip().lower()

                    if (e1.entity_id, e2.entity_id) in seen_pairs:
                        continue

                    # Exact duplicate
                    if s1 == s2:
                        issues.append(
                            QualityIssue(
                                issue_id=f"iss_dup_{uuid.uuid4().hex[:6]}",
                                type="duplicate",
                                field=etype,
                                current_value=e1.value,
                                conflicting_values=[e2.value],
                                evidence=f"Exact duplicate {etype} detected: '{e1.value}' (Page {e1.source_page} and Page {e2.source_page})",
                                severity="low",
                                source_page=e2.source_page,
                            )
                        )
                        seen_pairs.add((e1.entity_id, e2.entity_id))
                    else:
                        # Near-duplicate via RapidFuzz >= 85% or 1-edit distance
                        sim = fuzz.ratio(s1, s2)
                        edit_dist = abs(len(s1) - len(s2)) + sum(c1 != c2 for c1, c2 in zip(s1, s2))
                        if sim >= self.fuzzy_similarity_threshold or edit_dist <= 1:
                            issues.append(
                                QualityIssue(
                                    issue_id=f"iss_ndup_{uuid.uuid4().hex[:6]}",
                                    type="duplicate",
                                    field=etype,
                                    current_value=e1.value,
                                    conflicting_values=[e2.value],
                                    evidence=f"Near-duplicate {etype} ({sim:.1f}% similarity): '{e1.value}' vs '{e2.value}'",
                                    severity="medium",
                                    source_page=e2.source_page,
                                )
                            )
                            seen_pairs.add((e1.entity_id, e2.entity_id))

        # -------------------------------------------------------------------
        # 3. Format Inconsistent: Phone formats, Date formats, and Casing
        # -------------------------------------------------------------------
        # Inconsistent phones: e.g. raw digits vs hyphenated vs international prefix
        phone_patterns = set()
        for p_ent in by_type.get("Phone", []):
            val = p_ent.value.strip()
            if val.startswith("+91"):
                phone_patterns.add("+91")
            elif "-" in val:
                phone_patterns.add("hyphenated")
            elif val.isdigit():
                phone_patterns.add("raw_digits")

        if len(phone_patterns) > 1:
            for p_ent in by_type.get("Phone", []):
                if p_ent.normalized_value and p_ent.value != p_ent.normalized_value:
                    issues.append(
                        QualityIssue(
                            issue_id=f"iss_fmt_{uuid.uuid4().hex[:6]}",
                            type="format_inconsistent",
                            field="Phone",
                            current_value=p_ent.value,
                            conflicting_values=[p_ent.normalized_value],
                            evidence=f"Inconsistent phone format '{p_ent.value}'; standard E.164 is '{p_ent.normalized_value}'",
                            severity="low",
                            source_page=p_ent.source_page,
                        )
                    )

        # Inconsistent dates: e.g. DD/MM/YYYY vs YYYY-MM-DD
        date_patterns = set()
        for d_ent in by_type.get("Date", []):
            if "/" in d_ent.value:
                date_patterns.add("slash")
            elif "-" in d_ent.value:
                date_patterns.add("hyphen")
            elif "." in d_ent.value:
                date_patterns.add("dot")

        if len(date_patterns) > 1 or any(
            d.normalized_value != d.value for d in by_type.get("Date", [])
        ):
            for d_ent in by_type.get("Date", []):
                if d_ent.normalized_value and d_ent.value != d_ent.normalized_value:
                    issues.append(
                        QualityIssue(
                            issue_id=f"iss_datefmt_{uuid.uuid4().hex[:6]}",
                            type="format_inconsistent",
                            field="Date",
                            current_value=d_ent.value,
                            conflicting_values=[d_ent.normalized_value],
                            evidence=f"Inconsistent date format '{d_ent.value}'; ISO 8601 standard is '{d_ent.normalized_value}'",
                            severity="low",
                            source_page=d_ent.source_page,
                        )
                    )

        # -------------------------------------------------------------------
        # 4. Conflicting Values: Same subject having contradictory values
        # -------------------------------------------------------------------
        if graph and graph.clusters:
            for cluster in graph.clusters:
                name = cluster.get("primary_entity", "Unknown")
                attrs = cluster.get("attributes", {})
                for attr_type in ["Phone", "City", "State"]:
                    vals = set(attrs.get(attr_type, []))
                    if len(vals) > 1:
                        issues.append(
                            QualityIssue(
                                issue_id=f"iss_confl_{uuid.uuid4().hex[:6]}",
                                type="conflict",
                                field=attr_type,
                                current_value=list(vals)[0],
                                conflicting_values=list(vals)[1:],
                                evidence=f"Conflicting {attr_type} values for '{name}': {', '.join(vals)}",
                                severity="high",
                            )
                        )

        # -------------------------------------------------------------------
        # 5. Invalid: Regex / Domain validity
        # -------------------------------------------------------------------
        for em in by_type.get("Email", []):
            val = em.value.strip()
            if "@" not in val or "." not in val.split("@")[-1]:
                issues.append(
                    QualityIssue(
                        issue_id=f"iss_inv_{uuid.uuid4().hex[:6]}",
                        type="invalid",
                        field="Email",
                        current_value=val,
                        evidence=f"Malformed email address structure: '{val}'",
                        severity="high",
                        source_page=em.source_page,
                    )
                )

        for ph in by_type.get("Phone", []):
            digits = re.sub(r"\D", "", ph.value)
            if len(digits) < 10 or len(digits) > 15:
                issues.append(
                    QualityIssue(
                        issue_id=f"iss_inv_{uuid.uuid4().hex[:6]}",
                        type="invalid",
                        field="Phone",
                        current_value=ph.value,
                        evidence=f"Invalid phone number length ({len(digits)} digits): '{ph.value}'",
                        severity="high",
                        source_page=ph.source_page,
                    )
                )

        # -------------------------------------------------------------------
        # 6. Semantic: City & State variants (Hyd, HYD, Hyderabad -> Hyderabad)
        # -------------------------------------------------------------------
        city_variants = by_type.get("City", [])
        city_raw_vals = [c.value for c in city_variants]
        unique_city_low = {c.value.lower(): c.value for c in city_variants}

        # Check if there are multiple representations of the same canonical city
        canon_to_raws = defaultdict(set)
        for c_ent in city_variants:
            canon = c_ent.normalized_value or c_ent.value.title()
            canon_to_raws[canon].add(c_ent.value)

        for canon_city, raw_set in canon_to_raws.items():
            if len(raw_set) > 1:
                for raw_val in raw_set:
                    if raw_val != canon_city:
                        issues.append(
                            QualityIssue(
                                issue_id=f"iss_sem_{uuid.uuid4().hex[:6]}",
                                type="semantic",
                                field="City",
                                current_value=raw_val,
                                conflicting_values=[canon_city],
                                evidence=f"Semantic variant '{raw_val}' detected for canonical city '{canon_city}'",
                                severity="low",
                            )
                        )
            elif len(raw_set) == 1:
                # Even if only one raw value like "Hyd", if its canonical is "Hyderabad", flag it
                only_raw = next(iter(raw_set))
                if only_raw.lower() in ["hyd", "blr", "bombay", "calcutta", "madras"] and only_raw != canon_city:
                    issues.append(
                        QualityIssue(
                            issue_id=f"iss_sem_{uuid.uuid4().hex[:6]}",
                            type="semantic",
                            field="City",
                            current_value=only_raw,
                            conflicting_values=[canon_city],
                            evidence=f"Abbreviated/outdated city name '{only_raw}' maps to canonical '{canon_city}'",
                            severity="low",
                        )
                    )

        # State typos (e.g. Telengana -> Telangana)
        for s_ent in by_type.get("State", []):
            if s_ent.value.lower() == "telengana":
                issues.append(
                    QualityIssue(
                        issue_id=f"iss_sem_{uuid.uuid4().hex[:6]}",
                        type="semantic",
                        field="State",
                        current_value=s_ent.value,
                        conflicting_values=["Telangana"],
                        evidence=f"Spelling variation '{s_ent.value}' maps to canonical 'Telangana' (100% confidence)",
                        severity="low",
                        source_page=s_ent.source_page,
                    )
                )

        return issues

"""
L3 & L5: Constrained SLM Reasoning Planner & Confidence Governance Gate.

Features:
- Ingests profile (<2k tokens) + FDs + Semantic Types
- Prompts Qwen 0.5B via llama-cpp-python with GBNF grammar (temperature 0.1)
- Deterministic offline fallbacks with 100% GBNF schema compliance
- L5 Confidence Gate: routes steps >= 0.85 to auto_batch, < 0.85 to human_review_queue
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import polars as pl
from rapidfuzz import fuzz

from narvl.core.fd_miner import FunctionalDependency
from narvl.core.semantic_typer import (
    COMMON_CITIES,
    COMMON_INDIAN_STATES,
    SemanticClassification,
    US_STATES,
)
from narvl.engine.grammars import (
    CLEANING_PLAN_GBNF,
    GrammarValidationError,
    validate_cleaning_plan,
    validate_cleaning_step,
)
from narvl.engine.model_loader import resolve_model_path

logger = logging.getLogger("narvl.engine.planner")

CITY_ABBREVIATIONS: Dict[str, str] = {
    "hyd": "Hyderabad",
    "sec": "Secunderabad",
    "blr": "Bangalore",
    "mum": "Mumbai",
    "del": "Delhi",
    "chn": "Chennai",
    "kol": "Kolkata",
    "pune": "Pune",
    "nyc": "New York",
    "sf": "San Francisco",
    "la": "Los Angeles",
    "chi": "Chicago",
}

CONFIDENCE_GATE_THRESHOLD = 0.85


@dataclass
class PlanResult:
    """Outcome of SLM reasoning and L5 Confidence Gate routing."""
    steps: List[Dict[str, Any]]
    auto_batch: List[Dict[str, Any]]
    human_review_queue: List[Dict[str, Any]]
    raw_json: str
    is_valid_json: bool
    is_gbnf_compliant: bool


class SLMPlanner:
    """SLM Planner orchestrating deterministic data cleaning reasoning."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_GATE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.llama_model = None

        # Attempt to load local GGUF model if llama_cpp is installed
        try:
            import llama_cpp  # type: ignore
            resolved = resolve_model_path(allow_download=False)
            self.llama_model = llama_cpp.Llama(
                model_path=str(resolved),
                n_threads=4,    
                n_ctx=2048,
                mmap=True,
                verbose=False,
            )
            logger.info("Loaded llama.cpp local model from: %s", resolved)
        except Exception:
            self.llama_model = None

    def _generate_deterministic_plan(
        self,
        summary: Dict[str, Any],
        semantic_types: Dict[str, SemanticClassification],
        fds: List[FunctionalDependency],
        selected_columns: Optional[List[str]] = None,
        raw_df: Optional[pl.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Synthesize plan deterministically conforming strictly to GBNF schema."""
        steps: List[Dict[str, Any]] = []
        step_id = 1
        sel_set = set(selected_columns) if selected_columns is not None else None

        meta = summary.get("meta", {})
        dup_count = meta.get("duplicates", 0)
        dup_pct = meta.get("dup_pct", 0.0)
        columns_dict = summary.get("columns", {})
        has_id_col = any("id" in c.lower() or "key" in c.lower() or "code" in c.lower() for c in columns_dict)

        if dup_count > 0:
            if dup_pct > 15.0 and not has_id_col:
                # Without a unique identifier, high repetition indicates non-keyed transaction/event records.
                # Deduplicating would trigger catastrophic volumetric loss (>15%) and breach the safety gate.
                steps.append({
                    "step_id": step_id,
                    "target_column": "ALL",
                    "issue": f"High repetition ({dup_pct:.1f}%) detected without unique ID column",
                    "rule": "Preserve transaction event logs without identifier key",
                    "action": "deduplicate_exact",
                    "parameters": {"keep": "first"},
                    "confidence": 0.40,  # Routes to Human Review Queue (<0.85)
                    "justification": f"No primary key found; repeated feature tuples represent valid discrete events. Dropping would breach 15% Volumetric Loss Barrier.",
                    "loss_potential": "high",
                    "test_criterion": "verify_transaction_uniqueness",
                })
                step_id += 1
            else:
                steps.append({
                    "step_id": step_id,
                    "target_column": "ALL",
                    "issue": f"Detected {dup_count} exact duplicate rows",
                    "rule": "Row uniqueness constraint",
                    "action": "deduplicate_exact",
                    "parameters": {"keep": "first"},
                    "confidence": 0.99,
                    "justification": "Exact duplicates distort aggregate statistics and variance.",
                    "loss_potential": "low",
                    "test_criterion": "no_duplicate_primary_keys",
                })
                step_id += 1

        # 2. Check FDs for typo canonicalization and consolidate mappings
        fd_mappings_by_col: Dict[str, Dict[str, str]] = {}
        for fd in fds:
            if sel_set is not None and fd.dependent not in sel_set:
                continue
            if fd.canonical_mapping:
                if fd.dependent not in fd_mappings_by_col:
                    fd_mappings_by_col[fd.dependent] = {}
                fd_mappings_by_col[fd.dependent].update(fd.canonical_mapping)

        # 3. Semantic Entity Standardization (City, State, Phone)
        if raw_df is not None:
            for col in raw_df.columns:
                if sel_set is not None and col not in sel_set:
                    continue
                c_lower = col.lower()
                sem = semantic_types.get(col)
                sem_type_str = sem.semantic_type if sem else ""

                # City Standardization
                if "city" in c_lower or "town" in c_lower or sem_type_str == "City":
                    city_vals = [str(v).strip() for v in raw_df[col].drop_nulls().unique() if str(v).strip()]
                    c_map: Dict[str, str] = {}
                    for val in city_vals:
                        v_l = val.lower()
                        if v_l in CITY_ABBREVIATIONS:
                            can = CITY_ABBREVIATIONS[v_l]
                            if val != can:
                                c_map[val] = can
                        else:
                            for c_can in COMMON_CITIES:
                                canonical = c_can.title()
                                if len(v_l) >= 3 and c_can.startswith(v_l):
                                    if val != canonical:
                                        c_map[val] = canonical
                                    break
                                elif fuzz.ratio(v_l, c_can) >= 80:
                                    if val != canonical:
                                        c_map[val] = canonical
                                    break
                    if c_map:
                        if col not in fd_mappings_by_col:
                            fd_mappings_by_col[col] = {}
                        fd_mappings_by_col[col].update(c_map)

                # State Standardization
                if "state" in c_lower or "province" in c_lower or sem_type_str == "State":
                    state_vals = [str(v).strip() for v in raw_df[col].drop_nulls().unique() if str(v).strip()]
                    s_map: Dict[str, str] = {}
                    for val in state_vals:
                        v_l = val.lower()
                        for st_can in (COMMON_INDIAN_STATES | US_STATES):
                            canonical = st_can.title()
                            if v_l == st_can or fuzz.ratio(v_l, st_can) >= 80:
                                if val != canonical:
                                    s_map[val] = canonical
                                break
                    if s_map:
                        if col not in fd_mappings_by_col:
                            fd_mappings_by_col[col] = {}
                        fd_mappings_by_col[col].update(s_map)

                # Phone Standardization
                if "phone" in c_lower or "mobile" in c_lower or "contact" in c_lower or sem_type_str == "Phone":
                    phone_vals = [str(v).strip() for v in raw_df[col].drop_nulls().unique() if str(v).strip()]
                    p_map: Dict[str, Any] = {}
                    for val in phone_vals:
                        digits = re.sub(r"\D", "", str(val))
                        if len(digits) == 10:
                            canonical = f"+91-{digits}"
                            if str(val) != canonical:
                                p_map[str(val)] = canonical
                        elif len(digits) == 12 and digits.startswith("91"):
                            canonical = f"+91-{digits[2:]}"
                            if str(val) != canonical:
                                p_map[str(val)] = canonical
                        elif 0 < len(digits) < 10:
                            p_map[str(val)] = None
                    if p_map:
                        if col not in fd_mappings_by_col:
                            fd_mappings_by_col[col] = {}
                        fd_mappings_by_col[col].update(p_map)

        # Emit consolidated standardize_values steps with transitive closure
        for target_col, col_map in fd_mappings_by_col.items():
            if not col_map:
                continue
            # Transitive closure: if A -> B and B -> C, then A -> C
            resolved_map: Dict[str, Any] = {}
            for k, v in col_map.items():
                curr = v
                depth = 0
                while curr in col_map and depth < 5:
                    curr = col_map[curr]
                    depth += 1
                if k != curr:
                    resolved_map[k] = curr

            if resolved_map:
                steps.append({
                    "step_id": step_id,
                    "target_column": target_col,
                    "issue": f"Typographical, casing drift, and format anomalies in {target_col}",
                    "rule": f"Standardize {target_col} to canonical entities and formats",
                    "action": "standardize_values",
                    "parameters": {"mapping": resolved_map},
                    "confidence": 0.98,
                    "justification": f"Unified semantic knowledge and functional dependencies resolve {target_col} anomalies.",
                    "loss_potential": "none",
                    "test_criterion": f"standardized_{target_col.lower()}",
                })
                step_id += 1

        # 4. Check column profiles for nulls, bounds, and formatting
        columns = summary.get("columns", {})
        for col_name, col_meta in columns.items():
            if sel_set is not None and col_name not in sel_set:
                continue
            null_pct = col_meta.get("null_pct", 0.0)
            col_type = col_meta.get("type", "")

            # Ambiguous chaotic nulls (e.g. ~45% nulls)
            if 30.0 <= null_pct <= 60.0:
                steps.append({
                    "step_id": step_id,
                    "target_column": col_name,
                    "issue": f"High sparsity ({null_pct}% nulls) with ambiguous distribution",
                    "rule": "Information completeness threshold",
                    "action": "knn_impute",
                    "parameters": {"neighbors": 5, "strategy": "conditional_impute_or_drop"},
                    "confidence": 0.72,  # Sub-0.85 -> Routes to Human Review Queue
                    "justification": f"High missingness ({null_pct}%) introduces statistical bias; requires human sign-off.",
                    "loss_potential": "high",
                    "test_criterion": "null_percentage_under_10",
                })
                step_id += 1

            # Mild missingness (< 15%)
            elif 0.0 < null_pct < 15.0:
                steps.append({
                    "step_id": step_id,
                    "target_column": col_name,
                    "issue": f"Minor missing values ({null_pct}%)",
                    "rule": "Impute missing values using central tendency",
                    "action": "knn_impute",
                    "parameters": {"neighbors": 3, "strategy": "median" if "Int" in col_type or "Float" in col_type else "mode"},
                    "confidence": 0.92,  # Auto-batch
                    "justification": f"Low sparsity ({null_pct}%) can be safely imputed without statistical distortion.",
                    "loss_potential": "low",
                    "test_criterion": "non_null",
                })
                step_id += 1

            # Bounds validation on Age or similar numeric columns
            if "age" in col_name.lower() and "min" in col_meta and "max" in col_meta:
                if col_meta["min"] < 0 or col_meta["max"] > 120:
                    steps.append({
                        "step_id": step_id,
                        "target_column": col_name,
                        "issue": f"Age values out of plausible domain: [{col_meta['min']}, {col_meta['max']}]",
                        "rule": "Domain range: 0 <= Age <= 120",
                        "action": "clamp_bounds",
                        "parameters": {"lower": 0, "upper": 120},
                        "confidence": 0.98,
                        "justification": "Ages outside [0, 120] violate human biological domain boundaries.",
                        "loss_potential": "low",
                        "test_criterion": "between_0_and_120",
                    })
                    step_id += 1

            # Domain bounds for Financial / Amount columns (cannot be negative)
            if any(k in col_name.lower() for k in ["amount", "salary", "price", "revenue", "cost", "fee", "balance"]):
                min_v = col_meta.get("min")
                if min_v is None and raw_df is not None and col_name in raw_df.columns:
                    try:
                        clean_num = raw_df[col_name].cast(pl.Float64, strict=False).drop_nulls()
                        if len(clean_num) > 0:
                            min_v = float(clean_num.min())
                    except Exception:
                        pass
                if min_v is not None and min_v < 0:
                    max_v = float(col_meta.get("max", 1e9)) if col_meta.get("max") is not None else 1e9
                    steps.append({
                        "step_id": step_id,
                        "target_column": col_name,
                        "issue": f"Financial {col_name} values out of domain with negative values (min: {min_v})",
                        "rule": f"Domain range: {col_name} >= 0",
                        "action": "clamp_bounds",
                        "parameters": {"lower": 0.0, "upper": max_v},
                        "confidence": 0.98,
                        "justification": f"Financial amounts and salaries cannot be negative; clamp negative anomalies to zero.",
                        "loss_potential": "low",
                        "test_criterion": "non_negative",
                    })
                    step_id += 1

            # Regex replacement for formatting
            sem_type = semantic_types.get(col_name)
            if sem_type and sem_type.semantic_type == "Email":
                steps.append({
                    "step_id": step_id,
                    "target_column": col_name,
                    "issue": "Inconsistent email case, whitespace, or invalid format tokens",
                    "rule": "Email RFC standardization and validation",
                    "action": "regex_replace",
                    "parameters": {
                        "pattern": r"^\s+|\s+$",
                        "replacement": "",
                        "lowercase": True,
                        "nullify_invalid": True,
                        "valid_pattern": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
                    },
                    "confidence": 0.96,
                    "justification": "Emails must be trimmed, lowercased, and invalid format strings nullified.",
                    "loss_potential": "none",
                    "test_criterion": "valid_email_regex",
                })
                step_id += 1

        return steps

    def filter_plan_by_columns(
        self,
        steps: List[Dict[str, Any]],
        selected_columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter plan steps to only include actions targeting specified columns (or 'ALL')."""
        if selected_columns is None:
            return steps
        sel_set = set(selected_columns)
        return [
            s for s in steps
            if s.get("target_column") == "ALL" or s.get("target_column") in sel_set
        ]

    def plan(
        self,
        profile_summary: Union[str, Dict[str, Any]],
        semantic_types: Optional[Dict[str, SemanticClassification]] = None,
        fds: Optional[List[FunctionalDependency]] = None,
        selected_columns: Optional[List[str]] = None,
        raw_df: Optional[pl.DataFrame] = None,
    ) -> PlanResult:
        """Generate cleaning plan conforming to GBNF and execute L5 Confidence Gate.
        
        Args:
            profile_summary: Compact profile summary dict or JSON string.
            semantic_types: Optional mapping of column names to semantic classifications.
            fds: Optional list of discovered functional dependencies.
            selected_columns: Optional list of column names to constrain cleaning steps to.
            raw_df: Optional input DataFrame to derive semantic entity mappings from.
            
        Returns:
            PlanResult with steps routed to auto_batch and human_review_queue.
        """
        if isinstance(profile_summary, str):
            try:
                summary_dict = json.loads(profile_summary)
            except Exception:
                summary_dict = {}
        else:
            summary_dict = profile_summary

        semantic_types = semantic_types or {}
        fds = fds or []

        # If llama_model is available, run inference with GBNF grammar
        raw_output = ""
        steps: List[Dict[str, Any]] = []

        if self.llama_model is not None:
            try:
                prompt_text = (
                    f"Profile: {json.dumps(summary_dict)}\n"
                    f"Semantic Types: {[f'{k}:{v.semantic_type}' for k, v in semantic_types.items()]}\n"
                    f"FDs: {[f'{f.determinant}->{f.dependent}' for f in fds]}"
                )
                res = self.llama_model.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are NARVL Autonomous Data Cleaner. Output JSON list of steps.",
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    grammar=CLEANING_PLAN_GBNF,
                    temperature=0.1,
                    max_tokens=1024,
                )
                raw_output = res["choices"][0]["message"]["content"]
                parsed = json.loads(raw_output)
                if isinstance(parsed, list):
                    validate_cleaning_plan(parsed)
                    steps = parsed
            except Exception as e:
                logger.warning("Local llama-cpp inference failed or fell back: %s", e)
                steps = []

        # Deterministic constrained fallback reasoning
        if not steps:
            steps = self._generate_deterministic_plan(
                summary_dict, semantic_types, fds, selected_columns=selected_columns, raw_df=raw_df
            )
            raw_output = json.dumps(steps, indent=2)

        # Filter steps by selected_columns if specified
        if selected_columns is not None:
            steps = self.filter_plan_by_columns(steps, selected_columns)

        # Validate against GBNF schema
        validate_cleaning_plan(steps)

        # L5 Confidence Gate Routing
        auto_batch: List[Dict[str, Any]] = []
        human_review_queue: List[Dict[str, Any]] = []

        for step in steps:
            conf = float(step.get("confidence", 0.0))
            if conf >= self.confidence_threshold:
                auto_batch.append(step)
            else:
                human_review_queue.append(step)

        return PlanResult(
            steps=steps,
            auto_batch=auto_batch,
            human_review_queue=human_review_queue,
            raw_json=raw_output,
            is_valid_json=True,
            is_gbnf_compliant=True,
        )

    def generate_plan(
        self,
        profile_summary: Union[str, Dict[str, Any]],
        semantic_types: Optional[Dict[str, Any]] = None,
        functional_dependencies: Optional[List[Any]] = None,
        fds: Optional[List[FunctionalDependency]] = None,
        selected_columns: Optional[List[str]] = None,
        raw_df: Optional[pl.DataFrame] = None,
    ) -> PlanResult:
        """Convenience alias for plan with flexible dict and string type resolution."""
        actual_fds = fds or []
        if functional_dependencies and not actual_fds:
            for fd_item in functional_dependencies:
                if isinstance(fd_item, FunctionalDependency):
                    actual_fds.append(fd_item)
                elif isinstance(fd_item, dict):
                    actual_fds.append(
                        FunctionalDependency(
                            determinant=fd_item.get("determinant", ""),
                            dependent=fd_item.get("dependent", ""),
                            is_exact=fd_item.get("is_exact", True),
                            confidence=fd_item.get("confidence", 1.0),
                            canonical_mapping=fd_item.get("canonical_mapping", {}),
                        )
                    )
        converted_types: Dict[str, SemanticClassification] = {}
        if semantic_types:
            for k, v in semantic_types.items():
                if isinstance(v, SemanticClassification):
                    converted_types[k] = v
                elif isinstance(v, str):
                    converted_types[k] = SemanticClassification(
                        column_name=k,
                        semantic_type=v,
                        confidence=0.95,
                        raw_probabilities={v: 0.95},
                    )
                else:
                    converted_types[k] = v
        return self.plan(
            profile_summary=profile_summary,
            semantic_types=converted_types,
            fds=actual_fds,
            selected_columns=selected_columns,
            raw_df=raw_df,
        )

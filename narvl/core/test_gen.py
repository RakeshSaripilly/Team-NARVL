"""
L5: Dual Test Synthesizer for NARVL (Pandera + Great Expectations).

Auto-synthesizes blocking validation suites from cleaning plans:
1. Pandera DataFrameSchema (non-null PK, range bounds 0<=Age<=120, email regex)
2. Great Expectations expectation suite JSON file and executor
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pandera.polars as pa_pl
import polars as pl

EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"


@dataclass
class DualValidationResult:
    """Summary of Pandera and Great Expectations validation execution."""
    pandera_passed: bool
    pandera_errors: List[str]
    ge_passed: bool
    ge_summary: Dict[str, Any]
    is_fully_validated: bool
    removed_records_count: int = 0
    quarantined_df: Optional[pl.DataFrame] = None
    cleaned_df: Optional[pl.DataFrame] = None


class DualTestSynthesizer:
    """Synthesizes and runs blocking Pandera and Great Expectations test suites."""

    def __init__(self, plan_steps: Optional[List[Dict[str, Any]]] = None) -> None:
        self.plan_steps = plan_steps or []

    def build_pandera_schema(
        self,
        df: pl.DataFrame,
        plan_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> pa_pl.DataFrameSchema:
        """Construct Pandera DataFrameSchema from dataframe schema and cleaning plan."""
        steps = plan_steps if plan_steps is not None else self.plan_steps
        columns: Dict[str, pa_pl.Column] = {}

        # Scan plan steps for constraints
        col_constraints: Dict[str, Dict[str, Any]] = {}
        for s in steps:
            target = s.get("target_column")
            if not target or target == "ALL":
                continue
            if target not in col_constraints:
                col_constraints[target] = {}

            action = s.get("action")
            params = s.get("parameters", {})
            test_crit = s.get("test_criterion", "")

            if action == "clamp_bounds" or "between" in test_crit:
                col_constraints[target]["range"] = (params.get("lower", 0), params.get("upper", 120))
            if action == "knn_impute" or "non_null" in test_crit:
                col_constraints[target]["non_null"] = True
            if "email" in test_crit.lower() or action == "regex_replace":
                col_constraints[target]["email_regex"] = True

        for col_name in df.columns:
            checks = []
            c_info = col_constraints.get(col_name, {})
            dtype = df[col_name].dtype

            if "range" in c_info and dtype.is_numeric():
                low, high = c_info["range"]
                checks.append(pa_pl.Check.in_range(low, high))

            if "email_regex" in c_info and dtype in [pl.String, pl.Categorical]:
                checks.append(pa_pl.Check.str_matches(EMAIL_REGEX))

            nullable = not c_info.get("non_null", False)
            columns[col_name] = pa_pl.Column(checks=checks, nullable=nullable)

        return pa_pl.DataFrameSchema(columns)

    def build_ge_suite(
        self,
        df: pl.DataFrame,
        plan_steps: Optional[List[Dict[str, Any]]] = None,
        suite_name: str = "narvl_cleaning_suite",
    ) -> Dict[str, Any]:
        """Synthesize Great Expectations expectation suite dictionary."""
        steps = plan_steps if plan_steps is not None else self.plan_steps
        expectations: List[Dict[str, Any]] = []

        for s in steps:
            target = s.get("target_column")
            if not target or target == "ALL":
                continue
            action = s.get("action")
            params = s.get("parameters", {})
            test_crit = s.get("test_criterion", "")

            if action == "clamp_bounds" or "between" in test_crit:
                expectations.append({
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": target,
                        "min_value": params.get("lower", 0),
                        "max_value": params.get("upper", 120),
                    },
                })

            if action == "knn_impute" or "non_null" in test_crit:
                expectations.append({
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": target},
                })

            if "email" in test_crit.lower() or action == "regex_replace":
                expectations.append({
                    "expectation_type": "expect_column_values_to_match_regex",
                    "kwargs": {
                        "column": target,
                        "regex": EMAIL_REGEX,
                    },
                })

        return {
            "expectation_suite_name": suite_name,
            "expectations": expectations,
            "data_asset_type": "Dataset",
        }

    def validate_with_pandera(
        self,
        df: pl.DataFrame,
        schema: Optional[pa_pl.DataFrameSchema] = None,
        plan_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, List[str]]:
        """Run Pandera validation on Polars DataFrame with lazy row index collection."""
        if schema is None:
            schema = self.build_pandera_schema(df, plan_steps)

        errors: List[str] = []
        try:
            # Lazy validation collects all failure cases
            schema.validate(df, lazy=True)
            return True, []
        except Exception as exc:
            # Extract detailed schema errors
            err_msg = str(exc)
            errors.append(err_msg)
            return False, errors

    def validate_with_great_expectations(
        self,
        df: pl.DataFrame,
        suite: Optional[Dict[str, Any]] = None,
        plan_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run Great Expectations checks and return checkpoint verdict."""
        if suite is None:
            suite = self.build_ge_suite(df, plan_steps)

        all_passed = True
        results: List[Dict[str, Any]] = []

        for exp in suite.get("expectations", []):
            etype = exp["expectation_type"]
            kwargs = exp["kwargs"]
            col = kwargs.get("column")
            passed = True
            msg = ""

            if col not in df.columns:
                passed = False
                msg = f"Column '{col}' missing"
            else:
                series = df.get_column(col)
                if etype == "expect_column_values_to_be_between":
                    min_val = kwargs.get("min_value")
                    max_val = kwargs.get("max_value")
                    non_null = series.drop_nulls()
                    if len(non_null) > 0:
                        failed_count = int(((non_null < min_val) | (non_null > max_val)).sum())
                        if failed_count > 0:
                            passed = False
                            msg = f"Found {failed_count} values outside [{min_val}, {max_val}]"

                elif etype == "expect_column_values_to_not_be_null":
                    null_count = series.null_count()
                    if null_count > 0:
                        passed = False
                        msg = f"Found {null_count} unexpected null values"

                elif etype == "expect_column_values_to_match_regex":
                    pattern = kwargs.get("regex")
                    non_null = series.drop_nulls().cast(pl.String)
                    if len(non_null) > 0:
                        matched = non_null.str.contains(pattern)
                        failed_count = int((~matched).sum())
                        if failed_count > 0:
                            passed = False
                            msg = f"Found {failed_count} non-matching regex strings"

            if not passed:
                all_passed = False

            results.append({
                "expectation": etype,
                "column": col,
                "success": passed,
                "details": msg,
            })

        summary = {
            "success": all_passed,
            "total_expectations": len(results),
            "failed_expectations": sum(1 for r in results if not r["success"]),
            "results": results,
        }
        return all_passed, summary

    def filter_valid_records(
        self,
        df: pl.DataFrame,
        plan_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, int]:
        """Identify and remove records failing Pandera schema checks or Great Expectations.

        Returns:
            Tuple of (valid_df, quarantined_df, removed_count)
        """
        if df.height == 0:
            return df, df, 0

        steps = plan_steps if plan_steps is not None else self.plan_steps
        invalid_mask = pl.repeat(False, n=df.height, dtype=pl.Boolean)

        # 1. Evaluate checks from plan constraints (Range, Non-null, Regex)
        for s in steps:
            target = s.get("target_column")
            if not target or target == "ALL" or target not in df.columns:
                continue

            action = s.get("action")
            params = s.get("parameters", {})
            test_crit = s.get("test_criterion", "")

            # Range bounds check
            if action == "clamp_bounds" or "between" in test_crit:
                lower = params.get("lower", 0)
                upper = params.get("upper", 120)
                series = df.get_column(target)
                if series.dtype.is_numeric():
                    out_of_range = (series < lower) | (series > upper)
                    invalid_mask = invalid_mask | out_of_range.fill_null(False)

            # Non-null check
            if action == "knn_impute" or "non_null" in test_crit:
                series = df.get_column(target)
                invalid_mask = invalid_mask | series.is_null()

            # Email regex check
            if "email" in test_crit.lower() or action == "regex_replace":
                series = df.get_column(target)
                if series.dtype in [pl.String, pl.Categorical]:
                    str_s = series.cast(pl.String)
                    non_matching = str_s.is_not_null() & (~str_s.str.contains(EMAIL_REGEX))
                    invalid_mask = invalid_mask | non_matching.fill_null(False)

        # 2. Evaluate expectations from Great Expectations suite
        ge_suite = self.build_ge_suite(df, plan_steps=steps)
        for exp in ge_suite.get("expectations", []):
            etype = exp["expectation_type"]
            kwargs = exp["kwargs"]
            col = kwargs.get("column")
            if not col or col not in df.columns:
                continue
            series = df.get_column(col)

            if etype == "expect_column_values_to_be_between":
                min_v = kwargs.get("min_value")
                max_v = kwargs.get("max_value")
                if min_v is not None and max_v is not None and series.dtype.is_numeric():
                    out_b = (series < min_v) | (series > max_v)
                    invalid_mask = invalid_mask | out_b.fill_null(False)

            elif etype == "expect_column_values_to_not_be_null":
                invalid_mask = invalid_mask | series.is_null()

            elif etype == "expect_column_values_to_match_regex":
                pat = kwargs.get("regex", EMAIL_REGEX)
                if pat and series.dtype in [pl.String, pl.Categorical]:
                    str_s = series.cast(pl.String)
                    non_m = str_s.is_not_null() & (~str_s.str.contains(pat))
                    invalid_mask = invalid_mask | non_m.fill_null(False)

        valid_df = df.filter(~invalid_mask)
        quarantined_df = df.filter(invalid_mask)
        removed_count = quarantined_df.height

        return valid_df, quarantined_df, removed_count

    def filter_and_validate(
        self,
        df: pl.DataFrame,
        plan_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[pl.DataFrame, DualValidationResult]:
        """Filter out records failing Pandera or GE, and validate the resulting dataset."""
        valid_df, quarantined_df, removed_count = self.filter_valid_records(df, plan_steps=plan_steps)
        val_result = self.validate_dataset(valid_df, plan_steps=plan_steps)
        val_result.removed_records_count = removed_count
        val_result.quarantined_df = quarantined_df
        val_result.cleaned_df = valid_df
        return valid_df, val_result

    def validate_dataset(
        self,
        df: pl.DataFrame,
        plan_steps: Optional[List[Dict[str, Any]]] = None,
        filter_failing_records: bool = False,
    ) -> DualValidationResult:
        """Run both Pandera and Great Expectations suites simultaneously.

        If filter_failing_records is True, non-compliant rows are removed before validation.
        """
        target_df = df
        quarantined_df = None
        removed_count = 0

        if filter_failing_records:
            target_df, quarantined_df, removed_count = self.filter_valid_records(df, plan_steps=plan_steps)

        pan_passed, pan_errs = self.validate_with_pandera(target_df, plan_steps=plan_steps)
        ge_passed, ge_summary = self.validate_with_great_expectations(target_df, plan_steps=plan_steps)

        return DualValidationResult(
            pandera_passed=pan_passed,
            pandera_errors=pan_errs,
            ge_passed=ge_passed,
            ge_summary=ge_summary,
            is_fully_validated=(pan_passed and ge_passed),
            removed_records_count=removed_count,
            quarantined_df=quarantined_df,
            cleaned_df=target_df,
        )

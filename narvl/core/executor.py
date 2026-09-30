"""
L6: Reversible DAG Execution Engine for NARVL via Polars & Delta Lake (delta-rs).

Features:
- Fast vectorized transformations using Polars
- ACID Delta Lake commits for every executed step / pipeline version
- Full time-travel and instant rollback: DeltaTable.load_as_version(v)
- Complete transaction history and commit logs in _delta_log/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger("narvl.core.executor")


@dataclass
class ExecutionResult:
    """Outcome of DAG execution and Delta Lake commit."""
    initial_version: int
    final_version: int
    applied_steps: int
    total_steps: int
    cleaned_df: pl.DataFrame
    table_uri: str
    null_resolution: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Allow tuple unpacking (cleaned_df, report_dict) for backwards compatibility."""
        report = {
            "initial_version": self.initial_version,
            "final_version": self.final_version,
            "applied_steps": self.applied_steps,
            "total_steps": self.total_steps,
            "table_uri": self.table_uri,
            "null_resolution": self.null_resolution,
        }
        return iter((self.cleaned_df, report))


def resolve_nulls(
    df: pl.DataFrame,
    target_columns: Optional[List[str]] = None,
    semantic_types: Optional[Dict[str, Any]] = None,
    drop_unresolvable_rows: bool = True,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Replace null values with appropriate central tendency values (median/mode) where possible;
    otherwise remove rows containing unresolvable nulls.

    Rules:
    - Numeric features: Impute with median (if at least 1 non-null value exists).
    - Categorical/String features: Impute with mode (if at least 1 non-null value exists and not an ID/Email).
    - Boolean features: Impute with mode.
    - Identifier / Primary Key / Email columns: Cannot impute fabricated values -> remove rows.
    - 100% null columns: Cannot impute -> remove rows.
    - Remaining nulls: Purged row-wise if drop_unresolvable_rows is True.
    """
    if df.height == 0:
        return df, {"imputed": {}, "removed_rows": 0, "total_nulls_imputed": 0, "final_rows": 0}

    cols = [c for c in (target_columns if target_columns is not None else df.columns) if c in df.columns]
    imputed_report: Dict[str, Any] = {}
    curr_df = df

    for col in cols:
        ser = curr_df[col]
        null_count = ser.null_count()
        if null_count == 0:
            continue

        c_lower = col.lower()
        is_id = any(k in c_lower for k in ["id", "key", "code", "uuid", "ssn"])
        is_email = "email" in c_lower

        if semantic_types:
            sem = str(semantic_types.get(col, "")).lower()
            if "identifier" in sem or "id" in sem:
                is_id = True
            if "email" in sem:
                is_email = True

        non_null = ser.drop_nulls()
        if len(non_null) == 0 or is_id or is_email:
            continue

        fill_val = None
        strategy = ""
        if ser.dtype.is_numeric():
            median_val = float(non_null.median())
            strategy = "median"
            if ser.dtype.is_integer():
                fill_val = int(round(median_val))
            else:
                fill_val = float(median_val)
        elif ser.dtype == pl.Boolean:
            strategy = "mode"
            mode_vals = non_null.mode()
            fill_val = bool(mode_vals[0]) if len(mode_vals) > 0 else False
        elif ser.dtype in [pl.String, pl.Categorical]:
            strategy = "mode"
            mode_vals = non_null.mode()
            if len(mode_vals) > 0 and str(mode_vals[0]).strip() != "":
                fill_val = str(mode_vals[0])

        if fill_val is not None:
            curr_df = curr_df.with_columns(pl.col(col).fill_null(fill_val))
            imputed_report[col] = {
                "strategy": strategy,
                "fill_value": fill_val,
                "nulls_filled": null_count,
            }

    removed_rows = 0
    if drop_unresolvable_rows and cols:
        has_null = pl.any_horizontal([pl.col(c).is_null() for c in cols])
        filtered_df = curr_df.filter(~has_null)
        removed_rows = curr_df.height - filtered_df.height
        curr_df = filtered_df

    summary = {
        "imputed": imputed_report,
        "imputed_columns_count": len(imputed_report),
        "total_nulls_imputed": sum(v["nulls_filled"] for v in imputed_report.values()),
        "removed_rows": removed_rows,
        "final_rows": curr_df.height,
    }
    return curr_df, summary


class ReversibleExecutor:
    """ACID Reversible Pipeline Executor backed by Delta Lake Rust engine."""

    def __init__(
        self,
        table_uri: Union[str, Path, pl.DataFrame] = "data/delta_store",
        raw_df: Optional[pl.DataFrame] = None,
        delta_table_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if isinstance(table_uri, pl.DataFrame):
            self.raw_df = table_uri
            self.table_uri = str(delta_table_path or "data/delta_store")
        else:
            self.raw_df = raw_df
            self.table_uri = str(delta_table_path or table_uri)

    def initialize_store(self, raw_df: pl.DataFrame) -> int:
        """Write raw dataset as initial Delta commit version 0."""
        # Ensure target directory exists
        Path(self.table_uri).mkdir(parents=True, exist_ok=True)

        # Write version 0
        write_deltalake(
            self.table_uri,
            raw_df.to_arrow(),
            mode="overwrite",
        )
        dt = DeltaTable(self.table_uri)
        logger.info("Initialized Delta store at %s (version: %d)", self.table_uri, dt.version())
        return dt.version()

    def execute_step(self, df: pl.DataFrame, step: Dict[str, Any]) -> pl.DataFrame:
        """Apply a single transformation step via Polars."""
        action = step.get("action")
        target_col = step.get("target_column")
        params = step.get("parameters", {})

        if action == "deduplicate_exact":
            return df.unique()

        if action == "standardize_values" and target_col and target_col in df.columns:
            mapping = params.get("mapping", {})
            if mapping:
                return df.with_columns(pl.col(target_col).replace(mapping))
            return df

        if action == "clamp_bounds" and target_col and target_col in df.columns:
            low = params.get("lower", 0)
            high = params.get("upper", 120)
            return df.with_columns(pl.col(target_col).clip(low, high))

        if action == "knn_impute" and target_col and target_col in df.columns:
            ser = df[target_col]
            strategy = params.get("strategy", "median")
            if ser.dtype.is_numeric():
                fill_val = float(ser.drop_nulls().median()) if len(ser.drop_nulls()) > 0 else 0.0
            else:
                mode_vals = ser.drop_nulls().mode()
                fill_val = str(mode_vals[0]) if len(mode_vals) > 0 else ""
            return df.with_columns(pl.col(target_col).fill_null(fill_val))

        if action == "regex_replace" and target_col and target_col in df.columns:
            pattern = params.get("pattern", "")
            replacement = params.get("replacement", "")
            lowercase = params.get("lowercase", False)
            nullify_invalid = params.get("nullify_invalid", False) or params.get("nullify_non_matching", False)
            valid_pattern = params.get("valid_pattern")
            test_criterion = str(step.get("test_criterion", "")).lower()

            # Auto-detect email RFC validation if criterion or target indicates email regex
            if not valid_pattern and ("email" in test_criterion or "email" in str(target_col).lower()):
                valid_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
                nullify_invalid = True

            res_df = df
            if pattern:
                res_df = res_df.with_columns(
                    pl.col(target_col).cast(pl.String).str.replace_all(pattern, replacement)
                )
            if lowercase:
                res_df = res_df.with_columns(
                    pl.col(target_col).cast(pl.String).str.to_lowercase()
                )
            if (nullify_invalid or valid_pattern) and valid_pattern:
                res_df = res_df.with_columns(
                    pl.when(pl.col(target_col).str.contains(valid_pattern))
                    .then(pl.col(target_col))
                    .otherwise(None)
                    .alias(target_col)
                )
            return res_df

        return df

    def resolve_nulls(
        self,
        df: pl.DataFrame,
        target_columns: Optional[List[str]] = None,
        semantic_types: Optional[Dict[str, Any]] = None,
        drop_unresolvable_rows: bool = True,
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Replace nulls with median/mode where possible; otherwise remove rows with unresolvable nulls."""
        return resolve_nulls(
            df=df,
            target_columns=target_columns,
            semantic_types=semantic_types,
            drop_unresolvable_rows=drop_unresolvable_rows,
        )

    def execute_plan(
        self,
        raw_df: Union[pl.DataFrame, List[Dict[str, Any]]],
        plan_steps: Optional[List[Dict[str, Any]]] = None,
        commit_per_step: bool = False,
        filter_validation_failures: bool = False,
        target_columns: Optional[List[str]] = None,
        resolve_nulls_policy: bool = False,
        semantic_types: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute full DAG pipeline and commit to Delta Lake."""
        if isinstance(raw_df, list):
            actual_steps = raw_df
            actual_df = self.raw_df
            if actual_df is None:
                raise ValueError("raw_df must be supplied to execute_plan or __init__")
        else:
            actual_df = raw_df
            actual_steps = plan_steps or []

        # Scope steps to target_columns if specified
        if target_columns is not None:
            col_set = set(target_columns)
            actual_steps = [
                s for s in actual_steps
                if s.get("target_column") == "ALL" or s.get("target_column") in col_set
            ]

        init_v = self.initialize_store(actual_df)
        curr_df = actual_df
        applied_count = 0

        for step in actual_steps:
            curr_df = self.execute_step(curr_df, step)
            applied_count += 1
            if commit_per_step:
                write_deltalake(
                    self.table_uri,
                    curr_df.to_arrow(),
                    mode="overwrite",
                )

        # Null resolution policy: replace nulls where possible, remove rows where not possible
        null_res_report: Dict[str, Any] = {}
        if resolve_nulls_policy:
            curr_df, null_res_report = resolve_nulls(
                curr_df,
                target_columns=target_columns,
                semantic_types=semantic_types,
                drop_unresolvable_rows=True,
            )

        # Post-processing: optionally remove records failing Pandera or GE
        if filter_validation_failures and actual_steps:
            from narvl.core.test_gen import DualTestSynthesizer
            synth = DualTestSynthesizer(actual_steps)
            curr_df, _, _ = synth.filter_valid_records(curr_df, target_columns=target_columns)

        # Commit final state if modified
        had_post_filter = (resolve_nulls_policy and (null_res_report.get("total_nulls_imputed", 0) > 0 or null_res_report.get("removed_rows", 0) > 0)) or (filter_validation_failures and actual_steps)
        if not commit_per_step or applied_count == 0 or had_post_filter:
            write_deltalake(
                self.table_uri,
                curr_df.to_arrow(),
                mode="overwrite",
            )

        dt = DeltaTable(self.table_uri)
        final_v = dt.version()

        return ExecutionResult(
            initial_version=init_v,
            final_version=final_v,
            applied_steps=applied_count,
            total_steps=len(actual_steps),
            cleaned_df=curr_df,
            table_uri=self.table_uri,
            null_resolution=null_res_report,
        )

    def get_current_version(self) -> int:
        """Get the latest commit version of the Delta Lake table."""
        try:
            dt = DeltaTable(self.table_uri)
            return dt.version()
        except Exception:
            return 0

    def rollback_to_version(self, version: int) -> pl.DataFrame:
        """Time-travel back to a specific Delta version and restore Polars DataFrame."""
        dt = DeltaTable(self.table_uri)
        dt.load_as_version(version)
        arrow_table = dt.to_pyarrow_table()
        restored = pl.from_arrow(arrow_table)
        logger.info("Successfully rolled back %s to version %d", self.table_uri, version)
        return restored

    def get_commit_history(self) -> List[Dict[str, Any]]:
        """Retrieve full Delta Lake commit log history."""
        dt = DeltaTable(self.table_uri)
        return dt.history()

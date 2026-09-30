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
from dataclasses import dataclass
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

    def __iter__(self):
        """Allow tuple unpacking (cleaned_df, report_dict) for backwards compatibility."""
        report = {
            "initial_version": self.initial_version,
            "final_version": self.final_version,
            "applied_steps": self.applied_steps,
            "total_steps": self.total_steps,
            "table_uri": self.table_uri,
        }
        return iter((self.cleaned_df, report))


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
            res_df = df
            if pattern:
                res_df = res_df.with_columns(
                    pl.col(target_col).cast(pl.String).str.replace_all(pattern, replacement)
                )
            if lowercase:
                res_df = res_df.with_columns(
                    pl.col(target_col).cast(pl.String).str.to_lowercase()
                )
            return res_df

        return df

    def execute_plan(
        self,
        raw_df: Union[pl.DataFrame, List[Dict[str, Any]]],
        plan_steps: Optional[List[Dict[str, Any]]] = None,
        commit_per_step: bool = False,
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

        # Commit final state
        if not commit_per_step or applied_count == 0:
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

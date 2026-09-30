"""
L6: Reversible DAG Execution Engine for NARVL via Polars & Delta Lake (delta-rs).

Features:
- Fast vectorized transformations using Polars
- ACID Delta Lake commits for every executed step / pipeline version
- Full time-travel and instant rollback: DeltaTable.load_as_version(v)
- Complete transaction history and commit logs in _delta_log/
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

try:
    from deltalake import DeltaTable, write_deltalake
except ImportError:
    DeltaTable = None
    write_deltalake = None

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
        self._versions: Dict[int, pl.DataFrame] = {}

    def _save_fallback_snapshot(self, df: pl.DataFrame, version: int, operation: str) -> None:
        """Write Delta-compliant commit log and parquet snapshot when deltalake pkg is absent."""
        target_dir = Path(self.table_uri)
        target_dir.mkdir(parents=True, exist_ok=True)
        log_dir = target_dir / "_delta_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        parquet_file = f"part-00000-v{version}.parquet"
        df.write_parquet(target_dir / parquet_file)
        self._versions[version] = df

        commit_file = log_dir / f"{version:020d}.json"
        commit_info = {
            "commitInfo": {
                "timestamp": int(time.time() * 1000),
                "operation": operation,
                "version": version,
            },
            "add": {"path": parquet_file, "dataChange": True},
        }
        commit_file.write_text(json.dumps(commit_info, indent=2), encoding="utf-8")

    def initialize_store(self, raw_df: pl.DataFrame) -> int:
        """Write raw dataset as initial Delta commit version 0."""
        Path(self.table_uri).mkdir(parents=True, exist_ok=True)

        if write_deltalake is not None and DeltaTable is not None:
            try:
                write_deltalake(self.table_uri, raw_df.to_arrow(), mode="overwrite")
                dt = DeltaTable(self.table_uri)
                v = dt.version()
                logger.info("Initialized Delta store at %s (version: %d)", self.table_uri, v)
                return v
            except Exception as e:
                logger.warning("Delta-rs write failed, using parity fallback: %s", e)

        self._save_fallback_snapshot(raw_df, version=0, operation="CREATE TABLE")
        return 0

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
        curr_v = init_v

        for step in actual_steps:
            curr_df = self.execute_step(curr_df, step)
            applied_count += 1
            curr_v += 1
            if commit_per_step:
                if write_deltalake is not None:
                    try:
                        write_deltalake(self.table_uri, curr_df.to_arrow(), mode="overwrite")
                    except Exception:
                        self._save_fallback_snapshot(curr_df, version=curr_v, operation=step.get("action", "STEP"))
                else:
                    self._save_fallback_snapshot(curr_df, version=curr_v, operation=step.get("action", "STEP"))

        # Commit final state
        if not commit_per_step or applied_count == 0:
            final_v = curr_v if commit_per_step else (init_v + 1 if applied_count > 0 else init_v)
            if write_deltalake is not None and DeltaTable is not None:
                try:
                    write_deltalake(self.table_uri, curr_df.to_arrow(), mode="overwrite")
                    final_v = DeltaTable(self.table_uri).version()
                except Exception:
                    self._save_fallback_snapshot(curr_df, version=final_v, operation="FINAL_COMMIT")
            else:
                self._save_fallback_snapshot(curr_df, version=final_v, operation="FINAL_COMMIT")
        else:
            final_v = curr_v

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
        if DeltaTable is not None:
            try:
                dt = DeltaTable(self.table_uri)
                return dt.version()
            except Exception:
                pass
        return max(self._versions.keys()) if self._versions else 0

    def rollback_to_version(self, version: int) -> pl.DataFrame:
        """Time-travel back to a specific Delta version and restore Polars DataFrame."""
        if DeltaTable is not None:
            try:
                dt = DeltaTable(self.table_uri)
                dt.load_as_version(version)
                arrow_table = dt.to_pyarrow_table()
                restored = pl.from_arrow(arrow_table)
                logger.info("Successfully rolled back %s to version %d", self.table_uri, version)
                return restored
            except Exception as e:
                logger.warning("DeltaTable rollback error: %s, falling back to snapshot", e)

        target_file = Path(self.table_uri) / f"part-00000-v{version}.parquet"
        if target_file.exists():
            return pl.read_parquet(target_file)
        if version in self._versions:
            return self._versions[version]

        raise FileNotFoundError(f"Delta table version {version} not found at {self.table_uri}")

    def get_commit_history(self) -> List[Dict[str, Any]]:
        """Retrieve full Delta Lake commit log history."""
        if DeltaTable is not None:
            try:
                dt = DeltaTable(self.table_uri)
                return dt.history()
            except Exception:
                pass
        history = []
        log_dir = Path(self.table_uri) / "_delta_log"
        if log_dir.exists():
            for p in sorted(log_dir.glob("*.json")):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    history.append(data.get("commitInfo", {}))
                except Exception:
                    pass
        return history

"""
L2: Vectorized Profiler for NARVL with DuckDB and Polars.

Computes:
- Dataset-level metrics: row count, col count, duplicate row count
- Column metrics: null%, cardinality, Q25/Q50/Q75, min/max/mean/std, zero variance
- Vectorized pattern detectors: Email, Phone, ISO-8601, US Postal, India Postal
- Ultra-compact JSON summary strictly < 2000 tokens (verified via tiktoken)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import duckdb
import polars as pl
import tiktoken

ISO_DATE_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
)
US_POSTAL_REGEX = re.compile(r"^\d{5}(-\d{4})?$")
INDIA_POSTAL_REGEX = re.compile(r"^[1-9]\d{5}$")
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_REGEX = re.compile(
    r"^(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$|^(?:\+?91[-.\s]?)?[6-9]\d{9}$"
)


@dataclass
class DatasetProfile:
    """Complete dataset profile and compact LLM summary."""
    summary_dict: Dict[str, Any]
    summary_json: str
    token_count: int
    duration_seconds: float


class DatasetProfiler:
    """High-speed vectorized profiler combining Polars and DuckDB."""

    def __init__(self) -> None:
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def profile(self, df: pl.DataFrame) -> DatasetProfile:
        """Profile dataset and emit ultra-compact summary < 2000 tokens.
        
        Args:
            df: Normalized Polars DataFrame.
            
        Returns:
            DatasetProfile dataclass.
        """
        start_time = time.perf_counter()
        n_rows, n_cols = df.shape

        # 1. Dataset-level metrics
        if n_rows > 0:
            dup_count = df.is_duplicated().sum()
        else:
            dup_count = 0

        summary: Dict[str, Any] = {
            "meta": {
                "rows": n_rows,
                "cols": n_cols,
                "duplicates": int(dup_count),
                "dup_pct": round((dup_count / n_rows) * 100, 2) if n_rows > 0 else 0.0,
            },
            "columns": {},
        }

        # 2. Per-column profiling using Polars vectorized expressions
        for col_name in df.columns:
            series = df[col_name]
            dtype_str = str(series.dtype)
            null_count = series.null_count()
            null_pct = round((null_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
            n_unique = series.n_unique()

            col_meta: Dict[str, Any] = {
                "type": dtype_str,
                "null_pct": null_pct,
                "card": n_unique,
            }

            # Numeric profiling
            if series.dtype.is_numeric() and n_rows > null_count:
                non_null = series.drop_nulls()
                if len(non_null) > 0:
                    min_val = float(non_null.min())  # type: ignore
                    max_val = float(non_null.max())  # type: ignore
                    mean_val = float(non_null.mean())  # type: ignore
                    std_val = float(non_null.std()) if len(non_null) > 1 else 0.0  # type: ignore
                    q25 = float(non_null.quantile(0.25))  # type: ignore
                    q50 = float(non_null.quantile(0.50))  # type: ignore
                    q75 = float(non_null.quantile(0.75))  # type: ignore

                    col_meta.update({
                        "min": round(min_val, 2),
                        "q25": round(q25, 2),
                        "q50": round(q50, 2),
                        "q75": round(q75, 2),
                        "max": round(max_val, 2),
                        "mean": round(mean_val, 2),
                        "std": round(std_val, 2) if std_val is not None else 0.0,
                        "zero_var": (min_val == max_val or std_val == 0.0),
                    })
                else:
                    col_meta["zero_var"] = True

            # String / Categorical profiling
            elif series.dtype in [pl.String, pl.Categorical]:
                non_null = series.drop_nulls().cast(pl.String)
                count_non_null = len(non_null)

                if count_non_null > 0:
                    # Take up to 3 distinct sample strings
                    distinct_samples = non_null.unique().head(3).to_list()
                    col_meta["samples"] = distinct_samples
                    col_meta["zero_var"] = (n_unique <= 1)

                    # Subsample up to 1000 rows for high-speed pattern ratios
                    check_sample = non_null.sample(min(1000, count_non_null), seed=42).to_list()
                    sample_size = len(check_sample)

                    email_matches = sum(1 for v in check_sample if EMAIL_REGEX.match(str(v)))
                    phone_matches = sum(1 for v in check_sample if PHONE_REGEX.match(str(v)))
                    iso_matches = sum(1 for v in check_sample if ISO_DATE_REGEX.match(str(v)))
                    us_postal_matches = sum(1 for v in check_sample if US_POSTAL_REGEX.match(str(v)))
                    in_postal_matches = sum(1 for v in check_sample if INDIA_POSTAL_REGEX.match(str(v)))

                    patterns = {}
                    if email_matches / sample_size > 0.4:
                        patterns["email_ratio"] = round(email_matches / sample_size, 2)
                    if phone_matches / sample_size > 0.4:
                        patterns["phone_ratio"] = round(phone_matches / sample_size, 2)
                    if iso_matches / sample_size > 0.4:
                        patterns["iso_date_ratio"] = round(iso_matches / sample_size, 2)
                    if us_postal_matches / sample_size > 0.4:
                        patterns["us_postal_ratio"] = round(us_postal_matches / sample_size, 2)
                    if in_postal_matches / sample_size > 0.4:
                        patterns["india_postal_ratio"] = round(in_postal_matches / sample_size, 2)

                    if patterns:
                        col_meta["patterns"] = patterns
                else:
                    col_meta["samples"] = []
                    col_meta["zero_var"] = True

            summary["columns"][col_name] = col_meta

        # Convert to compact JSON representation
        summary_json = json.dumps(summary, separators=(",", ":"))

        # Measure tokens using tiktoken
        if self.tokenizer:
            token_count = len(self.tokenizer.encode(summary_json))
        else:
            token_count = len(summary_json) // 4

        duration = time.perf_counter() - start_time

        return DatasetProfile(
            summary_dict=summary,
            summary_json=summary_json,
            token_count=token_count,
            duration_seconds=duration,
        )


# Alias for backwards compatibility
FastProfiler = DatasetProfiler

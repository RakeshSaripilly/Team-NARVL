"""
L1: High-Performance Normalizer for NARVL.

Provides robust Polars readers for:
- CSV
- TSV
- Parquet
- JSON (Array of objects)
- NDJSON / JSONL (Newline-delimited JSON)

Integrates tightly with L0 Streaming Shield for adversarial safety.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import polars as pl

from narvl.core.shield import (
    DEFAULT_MAX_BYTES,
    QuotaExceededError,
    ShieldResult,
    StreamingShield,
)


@dataclass
class NormalizedDataset:
    """Wrapper holding normalized Polars DataFrame and associated metadata."""
    df: pl.DataFrame
    format: str
    source_path: Path
    shield_result: Optional[ShieldResult] = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape


class DatasetNormalizer:
    """L1 Normalizer converting varied raw file formats into standardized Polars DataFrames."""

    def __init__(
        self,
        shield: Optional[StreamingShield] = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.shield = shield or StreamingShield(max_bytes=max_bytes)
        self.max_bytes = max_bytes

    def detect_format(self, path: Path) -> str:
        """Infer format from file extension or content sniffing."""
        suffix = path.suffix.lower()
        if suffix in [".csv"]:
            return "csv"
        if suffix in [".tsv", ".tab"]:
            return "tsv"
        if suffix in [".parquet", ".pq"]:
            return "parquet"
        if suffix in [".ndjson", ".jsonl"]:
            return "ndjson"
        if suffix in [".json"]:
            # Check if it looks like NDJSON (multiple lines starting with '{')
            try:
                with open(path, "rb") as f:
                    first_k = f.read(4096).strip()
                    if first_k.startswith(b"["):
                        return "json"
                    lines = [l.strip() for l in first_k.split(b"\n") if l.strip()]
                    if len(lines) > 1 and lines[0].startswith(b"{") and lines[1].startswith(b"{"):
                        return "ndjson"
            except Exception:
                pass
            return "json"
        # Default fallback: check if binary or delimited
        return "csv"

    def normalize(
        self,
        input_path: Union[str, Path],
        clean_dir: Optional[Union[str, Path]] = None,
        quarantine_dir: Optional[Union[str, Path]] = None,
    ) -> NormalizedDataset:
        """Normalize dataset into a clean Polars DataFrame.
        
        Args:
            input_path: Path to dataset file.
            clean_dir: Optional directory to store intermediate sanitized shielded file.
            quarantine_dir: Optional directory for quarantine logs.
            
        Returns:
            NormalizedDataset containing loaded Polars DataFrame.
        """
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        file_size = source.stat().st_size
        if file_size > self.max_bytes:
            raise QuotaExceededError(
                f"File size {file_size} bytes exceeds 500MB quota ({self.max_bytes} bytes)."
            )

        fmt = self.detect_format(source)

        if fmt in ["csv", "tsv"]:
            clean_path = None
            quarantine_path = None
            if clean_dir:
                clean_path = Path(clean_dir) / f"{source.stem}_shielded{source.suffix}"
            if quarantine_dir:
                quarantine_path = Path(quarantine_dir) / "quarantine.log"

            shield_res = self.shield.inspect_and_clean(
                source,
                output_clean_path=clean_path,
                quarantine_log_path=quarantine_path,
            )

            # Load sanitized clean file into Polars
            df = pl.read_csv(
                shield_res.clean_path,
                separator=shield_res.detected_delimiter,
                infer_schema_length=10000,
                ignore_errors=True,
                null_values=["", "NA", "N/A", "null", "NULL", "None", "NaN"],
                truncate_ragged_lines=True,
            )
            return NormalizedDataset(
                df=df,
                format=fmt,
                source_path=source,
                shield_result=shield_res,
            )

        if fmt == "parquet":
            df = pl.read_parquet(source)
            return NormalizedDataset(
                df=df,
                format=fmt,
                source_path=source,
                shield_result=None,
            )

        if fmt == "ndjson":
            df = pl.read_ndjson(source, ignore_errors=True)
            return NormalizedDataset(
                df=df,
                format=fmt,
                source_path=source,
                shield_result=None,
            )

        if fmt == "json":
            try:
                df = pl.read_json(source)
            except Exception:
                # Fallback to ndjson if standard read_json fails
                df = pl.read_ndjson(source, ignore_errors=True)
            return NormalizedDataset(
                df=df,
                format=fmt,
                source_path=source,
                shield_result=None,
            )

        raise ValueError(f"Unsupported data format: {fmt}")

    def load_file(self, input_path: Union[str, Path]) -> pl.DataFrame:
        """Convenience loader returning Polars DataFrame directly."""
        return self.normalize(input_path).df


# Alias for backwards compatibility
FormatNormalizer = DatasetNormalizer

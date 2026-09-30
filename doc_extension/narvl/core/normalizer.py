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
            # Check if it looks like NDJSON (multiple lines each containing a full JSON object)
            try:
                with open(path, "rb") as f:
                    first_k = f.read(4096).strip()
                    if first_k.startswith(b"["):
                        return "json"
                    lines = [l.strip() for l in first_k.split(b"\n") if l.strip()]
                    if len(lines) > 1 and lines[0].startswith(b"{") and lines[0].endswith(b"}") and lines[1].startswith(b"{"):
                        return "ndjson"
            except Exception:
                pass
            return "json"
        # Default fallback: check if binary or delimited
        return "csv"

    def _load_ndjson_fallback(self, source: Path) -> pl.DataFrame:
        """Line-by-line fallback NDJSON reader using Python json parser."""
        records = []
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    if isinstance(obj, dict):
                        records.append(obj)
                    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                        records.extend(obj)
                except Exception:
                    continue
        if records:
            return pl.from_dicts(records)
        return pl.DataFrame()

    def _load_json(self, source: Path) -> pl.DataFrame:
        """Robustly load arbitrary JSON into a tabular Polars DataFrame."""
        # 1. Fast path: pl.read_json
        try:
            df = pl.read_json(source)
            if df.height > 0 and df.width > 0:
                # If a single row object contained a nested list of dicts, unpack it
                if df.height == 1:
                    for col in df.columns:
                        series = df[col]
                        if series.dtype == pl.List or isinstance(series[0], (list, pl.Series)):
                            sample_item = series[0]
                            if (isinstance(sample_item, list) and len(sample_item) > 0 and isinstance(sample_item[0], dict)) or (isinstance(sample_item, pl.Series) and sample_item.dtype == pl.Struct):
                                try:
                                    unwrapped = df.explode(col).unnest(col)
                                    if unwrapped.height > 1:
                                        return self._postprocess_dataframe(unwrapped)
                                except Exception:
                                    pass
                return self._postprocess_dataframe(df)
        except Exception:
            pass

        # 2. Parse using Python standard json with syntax repair
        data = None
        raw_text = ""
        try:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            data = json.loads(raw_text)
        except Exception:
            stripped = raw_text.strip()
            # Recover outer brace mistake: { { ... }, { ... } } -> [ { ... }, { ... } ]
            if stripped.startswith("{") and stripped.endswith("}"):
                inner = stripped[1:-1].strip()
                if inner.startswith("{"):
                    try:
                        data = json.loads(f"[{inner}]")
                    except Exception:
                        pass

            # Recover missing outer brackets: { ... }, { ... }
            if data is None:
                try:
                    data = json.loads(f"[{stripped}]")
                except Exception:
                    pass

            # Extract objects via raw_decode scan
            if data is None:
                decoder = json.JSONDecoder()
                pos = 0
                records = []
                text_to_scan = stripped
                if stripped.startswith("{") and stripped.find("{", 1) != -1:
                    first_inner = stripped[1:].lstrip()
                    if first_inner.startswith("{"):
                        text_to_scan = first_inner.rstrip("}")
                while pos < len(text_to_scan):
                    idx = text_to_scan.find("{", pos)
                    if idx == -1:
                        break
                    try:
                        obj, end = decoder.raw_decode(text_to_scan, idx)
                        if isinstance(obj, dict):
                            records.append(obj)
                        pos = end
                    except Exception:
                        pos = idx + 1
                if records:
                    data = records

            # Check if source is actually NDJSON
            if data is None:
                try:
                    return self._load_ndjson_fallback(source)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON file {source.name}: {e}")

        if isinstance(data, list):
            if not data:
                return pl.DataFrame()
            if isinstance(data[0], dict):
                return self._postprocess_dataframe(pl.from_dicts(data))
            return self._postprocess_dataframe(pl.DataFrame({"value": data}))

        if isinstance(data, dict):
            # Check for container keys: data, records, rows, items, results, entities, etc.
            for key in ["data", "records", "rows", "items", "results", "entities", "entries", "content"]:
                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                    if isinstance(data[key][0], dict):
                        return self._postprocess_dataframe(pl.from_dicts(data[key]))

            # Check if any key has list of dicts
            for key, val in data.items():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    return self._postprocess_dataframe(pl.from_dicts(val))

            # Check if columnar (dict of lists of same length)
            list_lens = [len(v) for v in data.values() if isinstance(v, list)]
            if list_lens and all(l == list_lens[0] for l in list_lens) and list_lens[0] > 0:
                try:
                    return self._postprocess_dataframe(pl.DataFrame(data))
                except Exception:
                    pass

            # Fallback: single record dictionary
            return self._postprocess_dataframe(pl.from_dicts([data]))

        raise ValueError(f"Unable to convert JSON of type {type(data).__name__} into a DataFrame.")

    def _postprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Coerce numeric strings to numbers, strip whitespace from text columns, and unnest struct columns."""
        if df.height == 0 or df.width == 0:
            return df

        # 0. Unnest struct columns (e.g. from nested JSON objects)
        struct_cols = [c for c in df.columns if isinstance(df[c].dtype, pl.Struct) or df[c].dtype == pl.Struct]
        for sc in struct_cols:
            try:
                field_names = [f.name for f in df[sc].dtype.fields]
                existing_other_cols = set(df.columns) - {sc}
                if any(fn in existing_other_cols for fn in field_names):
                    renamed_expr = pl.col(sc).struct.rename_fields([f"{sc}_{fn}" for fn in field_names])
                    df = df.with_columns(renamed_expr).unnest(sc)
                else:
                    df = df.unnest(sc)
            except Exception:
                pass

        for c in df.columns:
            if df[c].dtype == pl.String:
                c_lower = c.lower()
                is_code_or_id = any(k in c_lower for k in ["code", "zip", "postal", "phone", "id", "ssn", "ein", "tax", "mobile"])
                if not is_code_or_id:
                    # 1. Check if column represents numeric values stored as strings (e.g. "1250.50", "$500")
                    try:
                        clean_str = df[c].str.replace_all(r"[\$,]", "").str.strip_chars()
                        casted = clean_str.cast(pl.Float64, strict=False)
                        if casted.null_count() == df[c].null_count() and df[c].drop_nulls().len() > 0:
                            df = df.with_columns(casted.alias(c))
                            continue
                    except Exception:
                        pass

                # 2. Trim whitespace and convert whitespace-only strings to null
                try:
                    df = df.with_columns(
                        pl.when(pl.col(c).str.strip_chars().str.len_bytes() == 0)
                        .then(None)
                        .otherwise(pl.col(c).str.strip_chars())
                        .alias(c)
                    )
                except Exception:
                    pass

        return df

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
            try:
                df = pl.read_ndjson(source, ignore_errors=True)
            except Exception:
                df = self._load_ndjson_fallback(source)
            return NormalizedDataset(
                df=df,
                format=fmt,
                source_path=source,
                shield_result=None,
            )

        if fmt == "json":
            df = self._load_json(source)
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

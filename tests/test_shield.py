"""
Test 1.2: Streaming Shield & Adversarial Resilience Verification.

Validates that:
- Files exceeding 500MB raise QuotaExceededError
- Null bytes (\x00) are neutralized
- 500-commas delimiter bombs and ragged rows are safely isolated to quarantine.log
- 10MB ragged files parse with zero unhandled exceptions and yield valid Polars DataFrames
- Multi-tier model loader resolves tiers correctly
- Normalizer handles CSV, TSV, Parquet, JSON, and NDJSON
"""

import os
import shutil
from pathlib import Path

import polars as pl
import pytest

from narvl.core.normalizer import DatasetNormalizer, NormalizedDataset
from narvl.core.shield import (
    DEFAULT_MAX_BYTES,
    QuotaExceededError,
    ShieldResult,
    StreamingShield,
)
from narvl.engine.model_loader import (
    DEFAULT_MODEL_FILENAME,
    ModelResolutionError,
    resolve_model_path,
)


def test_streaming_quota_exceeded_600mb(tmp_path):
    """Test 1.2a: Stream 510MB synthetic CSV with null bytes and delimiter bomb.
    
    Must abort at 500MB ceiling with QuotaExceededError.
    """
    large_csv = tmp_path / "stream_510mb_synthetic.csv"
    
    # Generate 510 MB file rapidly using 8MB pre-computed blocks
    header = b"id,name,email,score,notes\n"
    bomb_row = b"9999," + (b"," * 500) + b"\n"
    regular_unit = b"1,Alice\x00Smith,alice@example.com,95.5,Regular note\n"
    
    # 8MB memory block
    block_size = 8 * 1024 * 1024
    block = (regular_unit * (block_size // len(regular_unit) + 1))[:block_size]

    target_bytes = 510 * 1024 * 1024  # 510 MB (> 500MB ceiling)
    
    with open(large_csv, "wb") as f:
        f.write(header)
        f.write(bomb_row)
        written = len(header) + len(bomb_row)
        while written < target_bytes:
            to_write = min(len(block), target_bytes - written)
            f.write(block[:to_write])
            written += to_write

    assert large_csv.stat().st_size >= 500 * 1024 * 1024

    shield = StreamingShield(max_bytes=DEFAULT_MAX_BYTES)
    clean_out = tmp_path / "clean_large.csv"
    quarantine_out = tmp_path / "quarantine_large.log"

    with pytest.raises(QuotaExceededError) as exc_info:
        shield.inspect_and_clean(
            input_path=large_csv,
            output_clean_path=clean_out,
            quarantine_log_path=quarantine_out,
        )

    error_msg = str(exc_info.value)
    print(f"\n[Shield Test 1.2a] Quota check passed: {error_msg}")
    assert "Cumulative 500MB ceiling exceeded" in error_msg

    # Cleanup large file immediately to conserve disk space
    large_csv.unlink(missing_ok=True)
    clean_out.unlink(missing_ok=True)
    quarantine_out.unlink(missing_ok=True)


def test_parse_10mb_ragged_file_with_delimiter_bombs(tmp_path):
    """Test 1.2b: Parse 10MB ragged file with delimiter bombs and null bytes.
    
    Verifies:
    - Bad rows (500-commas bomb, ragged lines) routed to quarantine.log
    - Null bytes stripped cleanly
    - Zero unhandled exceptions
    - Returns valid Polars DataFrame
    """
    ragged_csv = tmp_path / "ragged_10mb_test.csv"
    quarantine_log = tmp_path / "quarantine.log"
    clean_csv = tmp_path / "cleaned_data.csv"

    header = "customer_id,customer_name,email_address,credit_score,signup_date\n"
    valid_line = "1001,John Doe,john.doe@example.com,720,2023-01-15\n"
    null_byte_line = "1002,Jane\x00 Roe,jane.roe@example.com,810,2023-02-20\n"
    delimiter_bomb = "1003,Hacker Bomb," + ("," * 500) + ",MALICIOUS\n"
    ragged_short = "1004,Short Line\n"
    ragged_extra = "1005,Bob,bob@example.com,650,2023-03-10,extra_field_1,extra_field_2\n"

    target_bytes = 10 * 1024 * 1024  # 10 MB
    
    # 1MB block for fast generation
    block_1mb = (valid_line * (1024 * 1024 // len(valid_line) + 1)).encode("utf-8")[:1024*1024]

    with open(ragged_csv, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(delimiter_bomb.encode("utf-8"))
        f.write(null_byte_line.encode("utf-8"))
        f.write(ragged_short.encode("utf-8"))
        f.write(ragged_extra.encode("utf-8"))
        
        written = len(header) + len(delimiter_bomb) + len(null_byte_line) + len(ragged_short) + len(ragged_extra)
        while written < target_bytes:
            to_write = min(len(block_1mb), target_bytes - written)
            f.write(block_1mb[:to_write])
            written += to_write

    assert ragged_csv.stat().st_size >= 10 * 1024 * 1024

    # Run Normalizer (which exercises L0 StreamingShield and Polars loader)
    normalizer = DatasetNormalizer()
    normalized = normalizer.normalize(
        input_path=ragged_csv,
        clean_dir=tmp_path,
        quarantine_dir=tmp_path,
    )

    df = normalized.df
    shield_res = normalized.shield_result

    print(f"\n[Shield Test 1.2b] 10MB Ragged Dataset Results:")
    print(f"  Shape: {df.shape}")
    print(f"  Quarantined lines: {shield_res.quarantined_count}")
    print(f"  Valid rows: {shield_res.valid_rows_count}")
    print(f"  Null bytes detected: {shield_res.detected_null_bytes}")
    print(f"  Detected Delimiter: '{shield_res.detected_delimiter}'")
    print(f"  Detected Encoding: '{shield_res.detected_encoding}'")

    # Assertions
    assert isinstance(df, pl.DataFrame)
    assert df.height > 10000
    assert df.width == 5
    assert shield_res.quarantined_count >= 2  # Delimiter bomb + ragged lines isolated
    assert shield_res.detected_null_bytes is True
    assert quarantine_log.exists()

    # Verify quarantine log contents
    quarantine_content = quarantine_log.read_text(encoding="utf-8")
    assert "QUARANTINE" in quarantine_content
    assert "Delimiter Bomb" in quarantine_content

    # Verify no null byte in loaded DataFrame string representations
    names = df["customer_name"].to_list()
    assert all("\x00" not in str(n) for n in names)


def test_multi_tier_model_loader(tmp_path, monkeypatch):
    """Test 4-tier model resolution logic."""
    # Tier 1: NARVL_MODEL_DIR
    mock_model_dir = tmp_path / "custom_models"
    mock_model_dir.mkdir()
    fake_gguf = mock_model_dir / DEFAULT_MODEL_FILENAME
    fake_gguf.write_bytes(b"GGUF_MOCK_DATA")

    monkeypatch.setenv("NARVL_MODEL_DIR", str(mock_model_dir))
    resolved = resolve_model_path(allow_download=False)
    assert resolved == fake_gguf
    print(f"\n[Model Loader] Tier 1 Resolution verified: {resolved}")

    # Remove Tier 1 env
    monkeypatch.delenv("NARVL_MODEL_DIR", raising=False)

    # Tier 2: User Cache Directory
    mock_cache = tmp_path / ".cache" / "narvl" / "models"
    mock_cache.mkdir(parents=True)
    cache_gguf = mock_cache / DEFAULT_MODEL_FILENAME
    cache_gguf.write_bytes(b"GGUF_CACHE_DATA")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    resolved_tier2 = resolve_model_path(allow_download=False)
    assert resolved_tier2 == cache_gguf
    print(f"[Model Loader] Tier 2 Resolution verified: {resolved_tier2}")

    # Remove cache file and test missing resolution
    cache_gguf.unlink()
    with pytest.raises(ModelResolutionError):
        resolve_model_path(allow_download=False)
    print("[Model Loader] Air-gapped fallback / Resolution error check verified.")


def test_normalizer_multi_format(tmp_path):
    """Test L1 Normalizer across TSV, JSON, NDJSON, and Parquet."""
    normalizer = DatasetNormalizer()

    # TSV
    tsv_file = tmp_path / "sample.tsv"
    tsv_file.write_text("id\tcity\tzip\n1\tHyderabad\t500081\n2\tBengaluru\t560001\n", encoding="utf-8")
    ds_tsv = normalizer.normalize(tsv_file)
    assert ds_tsv.shape == (2, 3)
    assert ds_tsv.format == "tsv"

    # Parquet
    pq_file = tmp_path / "sample.parquet"
    ds_tsv.df.write_parquet(pq_file)
    ds_pq = normalizer.normalize(pq_file)
    assert ds_pq.shape == (2, 3)
    assert ds_pq.format == "parquet"

    # NDJSON
    ndjson_file = tmp_path / "sample.ndjson"
    ndjson_file.write_text('{"id": 1, "name": "A"}\n{"id": 2, "name": "B"}\n', encoding="utf-8")
    ds_ndjson = normalizer.normalize(ndjson_file)
    assert ds_ndjson.shape == (2, 2)
    assert ds_ndjson.format == "ndjson"

    # JSON array
    json_file = tmp_path / "sample.json"
    json_file.write_text('[{"id": 1, "val": 10}, {"id": 2, "val": 20}]', encoding="utf-8")
    ds_json = normalizer.normalize(json_file)
    assert ds_json.shape == (2, 2)
    assert ds_json.format == "json"

    # Multiline JSON object starting with '{'
    json_obj_file = tmp_path / "sample_obj.json"
    json_obj_file.write_text('{\n  "id": 101,\n  "name": "Alice",\n  "active": true\n}\n', encoding="utf-8")
    ds_obj = normalizer.normalize(json_obj_file)
    assert ds_obj.shape == (1, 3)
    assert ds_obj.format == "json"

    # JSON object with container key ("data" / "records")
    json_records_file = tmp_path / "sample_container.json"
    json_records_file.write_text('{\n  "status": "success",\n  "data": [\n    {"id": 1, "score": 90},\n    {"id": 2, "score": 95}\n  ]\n}', encoding="utf-8")
    ds_container = normalizer.normalize(json_records_file)
    assert ds_container.shape[0] == 2
    assert "score" in ds_container.df.columns
    assert ds_container.format == "json"
    print("\n[Normalizer Test] All multi-format readers verified (TSV, Parquet, NDJSON, JSON, Object JSON).")


def test_shield_sanitizes_json_starting_with_bracket_and_brace(tmp_path):
    """Test that StreamingShield sanitizes both array and object JSON without delimiter bombing or corruption."""
    shield = StreamingShield()
    normalizer = DatasetNormalizer()

    # 1. JSON file starting with '{' and null bytes
    dirty_json = tmp_path / "dirty_object.json"
    dirty_json.write_bytes(b'{\n  "doc_id": "doc_001",\n  "title": "Report\x00Title",\n  "version": 1\n}')

    clean_path, quarantined = shield.sanitize_file(dirty_json, quarantine_log=tmp_path / "quarantine.log")
    assert quarantined == 0
    df = normalizer.load_file(clean_path)
    assert df.shape == (1, 3)
    assert df["title"][0] == "ReportTitle"
    assert "\x00" not in df["title"][0]

    # 2. JSON file starting with '[' and multiline formatting
    array_json = tmp_path / "multiline_array.json"
    array_json.write_text('[\n  {\n    "item": "A",\n    "cost": 10\n  },\n  {\n    "item": "B",\n    "cost": 20\n  }\n]', encoding="utf-8")
    clean_arr, q_arr = shield.sanitize_file(array_json, quarantine_log=tmp_path / "quarantine.log")
    assert q_arr == 0
    df_arr = normalizer.load_file(clean_arr)
    assert df_arr.shape == (2, 2)


"""
Test 5.1 & 5.2: Reversible Executor, Dual Test Synthesizer & Provenance Audit Verification.

Validates that:
- Test 5.1:
  - Synthesizes Pandera + Great Expectations from cleaning plan
  - Injected negative Age triggers Pandera SchemaError and GE checkpoint failure explicitly
  - Valid cleaned dataset passes both suites
- Test 5.2:
  - Ingests raw data to Delta Lake v0
  - Applies 3-step DAG -> commits v1
  - Checks _delta_log contains 2 commits
  - Executes rollback(0) -> assert_frame_equal(raw_df, restored_df) == True (100% bitwise parity)
  - Emits cryptographically signed provenance JSON and HTML reports
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from narvl.core.executor import ReversibleExecutor
from narvl.core.loss import LossEstimator
from narvl.core.provenance import ProvenanceReporter
from narvl.core.test_gen import DualTestSynthesizer


def test_dual_test_synthesizer_pandera_and_ge_blocking():
    """Test 5.1: Pandera and Great Expectations dual synthesis and blocking on negative Age."""
    plan_steps = [
        {
            "step_id": 1,
            "target_column": "Age",
            "action": "clamp_bounds",
            "parameters": {"lower": 0, "upper": 120},
            "confidence": 0.98,
            "justification": "Domain bounds: 0 <= Age <= 120",
            "loss_potential": "low",
            "test_criterion": "between_0_and_120",
        },
        {
            "step_id": 2,
            "target_column": "Email",
            "action": "regex_replace",
            "parameters": {"pattern": r"^\s+|\s+$", "replacement": ""},
            "confidence": 0.95,
            "justification": "Valid RFC email format",
            "loss_potential": "none",
            "test_criterion": "valid_email_regex",
        },
    ]

    # 1. Valid cleaned dataset
    valid_df = pl.DataFrame({
        "ID": [1, 2, 3],
        "Age": [25, 45, 60],
        "Email": ["alice@corp.com", "bob@org.net", "charlie@gmail.com"],
    })

    synthesizer = DualTestSynthesizer(plan_steps)
    res_valid = synthesizer.validate_dataset(valid_df)

    print("\n[Dual Test Synthesizer Test 5.1 - Clean Data Run]:")
    print(f"  Pandera Passed: {res_valid.pandera_passed}")
    print(f"  Great Expectations Passed: {res_valid.ge_passed}")
    print(f"  Fully Validated: {res_valid.is_fully_validated}")

    assert res_valid.pandera_passed is True
    assert res_valid.ge_passed is True
    assert res_valid.is_fully_validated is True

    # 2. Injected Corrupted Dataset (Negative Age: -15, and invalid email)
    corrupted_df = pl.DataFrame({
        "ID": [1, 2, 3],
        "Age": [25, -15, 60],  # Negative Age injected!
        "Email": ["alice@corp.com", "INVALID_EMAIL_NO_AT", "charlie@gmail.com"],
    })

    res_corrupted = synthesizer.validate_dataset(corrupted_df)

    print("\n[Dual Test Synthesizer Test 5.1 - Corrupted Data Run (Negative Age Injected)]:")
    print(f"  Pandera Passed: {res_corrupted.pandera_passed} (Expected False)")
    print(f"  Pandera Failure Errors: {res_corrupted.pandera_errors[:1]}")
    print(f"  Great Expectations Passed: {res_corrupted.ge_passed} (Expected False)")
    print(f"  GE Failed Expectations: {res_corrupted.ge_summary['failed_expectations']}")

    # Assert explicit test blocking
    assert res_corrupted.pandera_passed is False
    assert len(res_corrupted.pandera_errors) > 0
    assert res_corrupted.ge_passed is False
    assert res_corrupted.ge_summary["failed_expectations"] >= 1
    assert res_corrupted.is_fully_validated is False


def test_delta_lake_3_step_dag_and_reversibility_rollback(tmp_path):
    """Test 5.2: Ingest v0, apply 3-step DAG -> v1, rollback(0) -> assert_frame_equal == True."""
    delta_uri = tmp_path / "delta_store"
    
    # Raw dataset (with duplicate, raw Age, typo City)
    raw_df = pl.DataFrame({
        "id": [1, 2, 3, 3],  # duplicate row
        "name": ["Alice", "Bob", "Charlie", "Charlie"],
        "age": [30, -10, 150, 150],  # out of bounds
        "city": ["Hyd", "Bengaluru", "Mumbai", "Mumbai"],
    })

    executor = ReversibleExecutor(table_uri=delta_uri)

    # 3-step DAG
    dag_steps = [
        {
            "step_id": 1,
            "target_column": "ALL",
            "action": "deduplicate_exact",
            "parameters": {},
            "confidence": 0.99,
            "justification": "Deduplicate exact rows",
            "loss_potential": "low",
            "test_criterion": "no_duplicate_rows",
        },
        {
            "step_id": 2,
            "target_column": "age",
            "action": "clamp_bounds",
            "parameters": {"lower": 0, "upper": 120},
            "confidence": 0.98,
            "justification": "Clamp age to 0-120",
            "loss_potential": "low",
            "test_criterion": "between_0_and_120",
        },
        {
            "step_id": 3,
            "target_column": "city",
            "action": "standardize_values",
            "parameters": {"mapping": {"Hyd": "Hyderabad"}},
            "confidence": 0.98,
            "justification": "Standardize Hyd to Hyderabad",
            "loss_potential": "none",
            "test_criterion": "canonical_city",
        },
    ]

    # Execute full pipeline
    exec_res = executor.execute_plan(raw_df, dag_steps)

    print(f"\n[Delta Lake Test 5.2 Execution Results]:")
    print(f"  Initial Version: {exec_res.initial_version} (v0)")
    print(f"  Final Version: {exec_res.final_version} (v1)")
    print(f"  Applied Steps: {exec_res.applied_steps}")
    print(f"  Cleaned Shape: {exec_res.cleaned_df.shape}")

    assert exec_res.initial_version == 0
    assert exec_res.final_version == 1

    # Verify commits in _delta_log
    delta_log_dir = delta_uri / "_delta_log"
    assert delta_log_dir.exists(), "_delta_log directory missing!"
    commit_files = list(delta_log_dir.glob("*.json"))
    print(f"  Delta Log Commit Files: {[f.name for f in commit_files]}")
    assert len(commit_files) >= 2  # 00000000000000000000.json and 00000000000000000001.json

    history = executor.get_commit_history()
    print(f"  Delta History Length: {len(history)} commits")
    assert len(history) >= 2

    # Verify cleaned data transformations
    clean_df = exec_res.cleaned_df
    assert clean_df.height == 3  # Duplicate removed
    assert clean_df["age"].min() >= 0
    assert clean_df["age"].max() <= 120
    assert "Hyderabad" in clean_df["city"].to_list()
    assert "Hyd" not in clean_df["city"].to_list()

    # Time-Travel Rollback to Version 0
    restored_df = executor.rollback_to_version(0)

    print(f"\n[Delta Lake Rollback Verification]:")
    print(f"  Restored Shape: {restored_df.shape}")
    print(f"  Raw Shape: {raw_df.shape}")

    # Strict parity verification
    assert_frame_equal(raw_df, restored_df)
    print("  assert_frame_equal(raw_df, restored_df) == True (100% BITWISE PARITY RESTORED)")

    # Test Provenance Report Emission
    estimator = LossEstimator()
    loss_report = estimator.evaluate_loss_and_safety(raw_df, clean_df)
    synthesizer = DualTestSynthesizer(dag_steps)
    validation_res = synthesizer.validate_dataset(clean_df)

    reporter = ProvenanceReporter()
    manifest = reporter.create_manifest(
        raw_df=raw_df,
        cleaned_df=clean_df,
        steps=dag_steps,
        loss_report=loss_report,
        validation_result=validation_res,
        dataset_name="customers_test",
    )

    json_report = reporter.emit_json(manifest, tmp_path / "narvl_provenance_report.json")
    html_report = reporter.emit_html(manifest, tmp_path / "narvl_provenance_report.html")

    assert json_report.exists()
    assert html_report.exists()
    assert "digital_signature" in manifest
    print(f"  Signed Provenance JSON emitted: {json_report}")
    print(f"  Executive Provenance HTML emitted: {html_report}")

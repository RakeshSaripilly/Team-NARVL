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


def test_dual_test_synthesizer_record_filtering():
    """Test 5.1b: Verification that records failing Pandera or GE are cleanly removed."""
    plan_steps = [
        {
            "step_id": 1,
            "target_column": "Age",
            "action": "clamp_bounds",
            "parameters": {"lower": 0, "upper": 120},
            "test_criterion": "between_0_and_120",
        },
        {
            "step_id": 2,
            "target_column": "Email",
            "action": "regex_replace",
            "test_criterion": "valid_email_regex",
        },
    ]

    corrupted_df = pl.DataFrame({
        "ID": [1, 2, 3, 4],
        "Age": [25, -15, 60, 200],  # Rows 2 and 4 have invalid Age (-15, 200)
        "Email": ["alice@corp.com", "INVALID_EMAIL", "charlie@gmail.com", "dave@corp.com"],
    })

    synthesizer = DualTestSynthesizer(plan_steps)
    cleaned_df, val_res = synthesizer.filter_and_validate(corrupted_df)

    # Verify that invalid rows were removed
    assert val_res.removed_records_count == 2
    assert cleaned_df.height == 2
    assert cleaned_df["ID"].to_list() == [1, 3]

    # Verify that the filtered dataset passes 100% of checks
    assert val_res.pandera_passed is True
    assert val_res.ge_passed is True
    assert val_res.is_fully_validated is True


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


def test_executor_with_validation_filtering_and_ge_export(tmp_path):
    """Test 5.3: Executor filtering of validation failures and GE suite JSON export."""
    delta_uri = tmp_path / "delta_store_filtered"
    raw_df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [28, -5, 45],  # Row 2 has invalid age (-5)
    })

    plan_steps = [
        {
            "step_id": 1,
            "target_column": "age",
            "action": "clamp_bounds",
            "parameters": {"lower": 0, "upper": 120},
            "test_criterion": "between_0_and_120",
        },
    ]

    # Corrupt executor step intentionally or execute with filter_validation_failures=True
    executor = ReversibleExecutor(table_uri=delta_uri)
    
    # Ingest without step 1 clamp to simulate a record that violates bounds
    synth = DualTestSynthesizer(plan_steps)
    cleaned_df, val_res = synth.filter_and_validate(raw_df)
    assert val_res.removed_records_count == 1
    assert cleaned_df.height == 2
    assert cleaned_df["id"].to_list() == [1, 3]

    # Test GE suite export
    ge_file = tmp_path / "narvl_cleaning_suite.json"
    exported_path = synth.export_ge_suite(ge_file, df=raw_df)
    assert exported_path.exists()
    import json
    suite_data = json.loads(exported_path.read_text(encoding="utf-8"))
    assert "expectations" in suite_data
    assert suite_data["expectation_suite_name"] == "narvl_cleaning_suite"


def test_resolve_nulls_imputes_or_removes_unresolvable():
    """Test 5.4: resolve_nulls replaces nulls where possible and purges unresolvable records."""
    from narvl.core.executor import resolve_nulls

    df = pl.DataFrame({
        "ID": [101, 102, None, 104, 105],                  # ID is null in row 2 -> cannot impute ID!
        "Age": [20.0, 30.0, 40.0, None, 50.0],             # Age is null in row 3 -> numeric -> median is 35.0!
        "Department": ["HR", "IT", "Finance", "IT", None],  # Dept is null in row 4 -> mode is "IT"!
        "Email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", None],  # Email is null in row 4 -> cannot impute Email!
    })

    cleaned_df, summary = resolve_nulls(df)

    # Imputed Age with median (35.0)
    assert "Age" in summary["imputed"]
    assert summary["imputed"]["Age"]["strategy"] == "median"
    assert summary["imputed"]["Age"]["fill_value"] == 35.0

    # Imputed Department with mode ("IT")
    assert "Department" in summary["imputed"]
    assert summary["imputed"]["Department"]["strategy"] == "mode"
    assert summary["imputed"]["Department"]["fill_value"] == "IT"

    # ID null (row 2) and Email null (row 4) cannot be imputed -> 2 rows removed
    assert summary["removed_rows"] == 2
    assert cleaned_df.height == 3
    assert cleaned_df["ID"].to_list() == [101, 102, 104]
    assert cleaned_df["Age"].to_list() == [20.0, 30.0, 35.0]
    assert cleaned_df["Department"].to_list() == ["HR", "IT", "IT"]
    assert cleaned_df["Email"].to_list() == ["a@x.com", "b@x.com", "d@x.com"]

    # Verify 0 nulls remain across entire dataframe
    assert cleaned_df.null_count().sum_horizontal()[0] == 0


def test_executor_column_selection_and_null_resolution(tmp_path):
    """Test 5.5: ReversibleExecutor targets specified columns and executes null resolution policy."""
    delta_uri = tmp_path / "delta_store_cols"

    raw_df = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "age": [25, 45, None, 180],
        "salary": [50000.0, None, 75000.0, 80000.0],
    })

    plan_steps = [
        {
            "step_id": 1,
            "target_column": "age",
            "action": "clamp_bounds",
            "parameters": {"lower": 0, "upper": 120},
            "test_criterion": "between_0_and_120",
        },
        {
            "step_id": 2,
            "target_column": "salary",
            "action": "clamp_bounds",
            "parameters": {"lower": 10000, "upper": 200000},
            "test_criterion": "between_10000_and_200000",
        },
    ]

    executor = ReversibleExecutor(table_uri=delta_uri)

    # Process ONLY 'age', leave 'salary' untouched
    res = executor.execute_plan(
        raw_df,
        plan_steps=plan_steps,
        target_columns=["age"],
        resolve_nulls_policy=True,
    )

    # Age was clamped (180 -> 120) and null was replaced with median (45)
    assert res.cleaned_df["age"].max() == 120
    assert res.cleaned_df["age"].null_count() == 0

    # Salary was NOT in target_columns, so its null is preserved (not dropped or imputed)
    assert res.cleaned_df["salary"].null_count() == 1
    assert res.cleaned_df.height == 4



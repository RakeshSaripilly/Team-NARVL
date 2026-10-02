"""
NARVL Command Line Interface (CLI).
Provides autonomous data cleaning pipeline commands and UI launcher.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import polars as pl

from narvl.core.executor import ReversibleExecutor
from narvl.core.fd_miner import FunctionalDependencyMiner
from narvl.core.loss import LossEstimator
from narvl.core.normalizer import FormatNormalizer
from narvl.core.profiler import FastProfiler
from narvl.core.provenance import ProvenanceReporter
from narvl.core.semantic_typer import SemanticTyper
from narvl.core.shield import StreamingShield
from narvl.core.test_gen import DualTestSynthesizer
from narvl.engine.planner import SLMPlanner


def run_clean_pipeline(
    input_file: Path,
    output_file: Optional[Path] = None,
    delta_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    auto_approve: bool = False,
    columns: Optional[str] = None,
    resolve_nulls: bool = True,
) -> int:
    """Execute end-to-end agentic cleaning pipeline on the input dataset."""
    selected_columns = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
    print(f"\n=======================================================")
    print(f"[*] NARVL: Autonomous Agentic Data Cleaning Engine")
    print(f"=======================================================")
    print(f"[*] Input dataset: {input_file}")
    if selected_columns:
        print(f"[*] Target columns to process: {', '.join(selected_columns)}")
    print(f"[*] Auto-resolve nulls policy: {resolve_nulls}")

    if not input_file.exists():
        print(f"[ERROR] File does not exist: {input_file}", file=sys.stderr)
        return 1

    # Default output paths
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
    if delta_dir is None:
        delta_dir = input_file.parent / f"{input_file.stem}_delta"
    if report_dir is None:
        report_dir = input_file.parent

    # 1. L0 Streaming Shield & Normalizer
    print("\n[Stage 1/7] Ingesting through L0 Streaming Shield & Format Normalizer...")
    shield = StreamingShield()
    quarantine_log = report_dir / "quarantine.log"
    sanitized_path, quarantined = shield.sanitize_file(input_file, quarantine_log=quarantine_log)
    if quarantined > 0:
        print(f"  [!] Isolated {quarantined} ragged rows to {quarantine_log}")

    normalizer = FormatNormalizer()
    raw_df = normalizer.load_file(sanitized_path)
    print(f"  [+] Ingested {raw_df.height} rows x {raw_df.width} columns")

    # 2. L1 Fast Profiler
    print("\n[Stage 2/7] Running L1 Vectorized Profiler...")
    profiler = FastProfiler()
    profile_res = profiler.profile(raw_df)
    profile_summary = profile_res.summary_dict
    token_count = profile_res.token_count
    print(f"  [+] Profile summary generated: {token_count} tokens (Budget < 2000 tokens)")

    # 3. L2 Semantic Typer & Approximate FD Miner
    print("\n[Stage 3/7] Inferring L2.5 Semantic Types & Functional Dependencies...")
    typer = SemanticTyper()
    types_found = typer.infer_types(raw_df)
    print(f"  [+] Inferred {len(types_found)} semantic types:")
    for col, pred in types_found.items():
        print(f"      - {col}: {pred.predicted_type} ({pred.confidence*100:.1f}%)")

    miner = FunctionalDependencyMiner(fuzzy_threshold=85.0, semantic_types=types_found)
    fds = miner.mine(raw_df, semantic_types=types_found)
    print(f"  [+] Discovered {len(fds)} functional dependencies:")
    for fd in fds:
        print(f"      - {fd.determinant} -> {fd.dependent} (Confidence: {fd.confidence:.2f})")

    # 4. L3 Constrained SLM Reasoning
    print("\n[Stage 4/7] Generating GBNF-Constrained Cleaning DAG...")
    planner = SLMPlanner()
    plan = planner.generate_plan(
        profile_summary=profile_summary,
        functional_dependencies=[fd.__dict__ for fd in fds],
        semantic_types={k: v.predicted_type for k, v in types_found.items()},
        selected_columns=selected_columns,
    )
    print(f"  [+] Inferred {len(plan.steps)} pipeline steps:")
    for s in plan.steps:
        tag = "[AUTO]" if s.get("confidence", 0) >= 0.85 else "[REVIEW]"
        print(f"      {tag} Step {s.get('step_id')}: {s.get('action')} on {s.get('target_column')} (conf: {s.get('confidence'):.2f})")

    # 5. L4 4D Loss Estimator & Speculative Utility Barrier
    print("\n[Stage 5/7] Simulating 4D Information Loss & Speculative Utility Barrier...")
    executor = ReversibleExecutor(raw_df, delta_table_path=delta_dir)
    candidate_df, _ = executor.execute_plan(
        plan.steps,
        target_columns=selected_columns,
        resolve_nulls_policy=resolve_nulls,
        dry_run=True,
    )

    loss_estimator = LossEstimator()
    assessment = loss_estimator.assess(raw_df, candidate_df)
    print(f"  [+] Volumetric Loss: {assessment.volumetric_loss*100:.2f}% (Limit: <=15%)")
    print(f"  [+] Statistical W1:  {assessment.statistical_w1:.4f} (Limit: <=0.35)")
    print(f"  [+] Jaccard Loss:    {assessment.categorical_jaccard_loss:.4f} (Limit: <=0.30)")
    print(f"  [+] Cosine Drift:    {assessment.semantic_cosine_drift:.4f} (Limit: <=0.20)")
    print(f"  [+] Utility Delta:   {assessment.predictive_utility_delta:+.4f} (Limit: >=0.0)")

    if not assessment.is_safe:
        print(f"\n[BLOCKED] 4D Loss / Utility Safety Gate Violation:")
        for r in assessment.blocking_reasons:
            print(f"  - {r}")
        if not auto_approve:
            print("[ABORT] Pipeline halted. Use --auto-approve to force override.", file=sys.stderr)
            return 2

    # 6. L5 Dual Test Synthesizer (Pandera + Great Expectations)
    print("\n[Stage 6/7] Synthesizing & Executing Pandera + Great Expectations Suites...")
    test_gen = DualTestSynthesizer(plan.steps)
    val_result = test_gen.validate_dataset(candidate_df, target_columns=selected_columns)
    print(f"  [+] Pandera Schema Passed: {val_result.pandera_passed}")
    print(f"  [+] Great Expectations Passed: {val_result.ge_passed}")

    if not val_result.is_fully_validated:
        print("  [!] Validation warnings encountered:")
        for err in val_result.pandera_errors:
            print(f"      {err}")

    # 7. Commit & Provenance Audit Report
    print("\n[Stage 7/7] Committing to Delta Lake & Signing Provenance Audit...")
    cleaned_df, report = executor.execute_plan(
        plan.steps,
        target_columns=selected_columns,
        resolve_nulls_policy=resolve_nulls,
        filter_validation_failures=True,
    )

    # Post-processing: Remove records failing Pandera or Great Expectations validation
    cleaned_df, val_result = test_gen.filter_and_validate(cleaned_df, target_columns=selected_columns)
    null_info = report.get("null_resolution", {})
    if null_info.get("total_nulls_imputed", 0) > 0 or null_info.get("removed_rows", 0) > 0:
        print(f"  [+] Null Policy: Imputed {null_info.get('total_nulls_imputed', 0)} null(s), removed {null_info.get('removed_rows', 0)} unresolvable record(s).")
    if val_result.removed_records_count > 0:
        print(f"  [!] Post-processing: Removed {val_result.removed_records_count} record(s) failing Pandera / GE validation.")

    output_ext = output_file.suffix.lower()
    if output_ext == ".parquet":
        cleaned_df.write_parquet(output_file)
    else:
        cleaned_df.write_csv(output_file)

    print(f"  [+] Saved cleaned dataset to: {output_file}")

    provenance = ProvenanceReporter()
    rep_json = report_dir / "narvl_provenance_report.json"
    rep_html = report_dir / "narvl_provenance_report.html"
    provenance.generate_report(
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        plan_steps=plan.steps,
        loss_assessment=assessment.__dict__,
        validation_result=val_result.__dict__,
        output_json=rep_json,
        output_html=rep_html,
    )
    print(f"  [+] Generated signed provenance JSON: {rep_json}")
    print(f"  [+] Generated executive HTML report:  {rep_html}")
    print("\n[SUCCESS] NARVL cleaning pipeline completed successfully.")
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Parse CLI arguments and dispatch commands."""
    parser = argparse.ArgumentParser(
        prog="narvl",
        description="NARVL: Autonomous, Offline, Plug-and-Play Agentic Data Cleaning Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: clean
    clean_parser = subparsers.add_parser("clean", help="Clean a dataset autonomously")
    clean_parser.add_argument("dataset", type=Path, help="Path to input dataset file (CSV, TSV, Parquet, JSON, NDJSON)")
    clean_parser.add_argument("-o", "--output", type=Path, default=None, help="Output cleaned file path")
    clean_parser.add_argument("--delta-dir", type=Path, default=None, help="Directory to store Delta Lake table")
    clean_parser.add_argument("--report-dir", type=Path, default=None, help="Directory to store provenance reports")
    clean_parser.add_argument("--auto-approve", action="store_true", help="Auto-approve steps despite safety gate warnings")
    clean_parser.add_argument("--columns", type=str, default=None, help="Comma-separated list of target columns to process (e.g. Age,Salary,Email)")
    clean_parser.add_argument("--no-resolve-nulls", dest="resolve_nulls", action="store_false", default=True, help="Disable automated null replacement/removal")

    # Command: ui
    ui_parser = subparsers.add_parser("ui", help="Launch Streamlit CleanPilot Web UI")
    ui_parser.add_argument("-p", "--port", type=int, default=8501, help="Port to bind Streamlit server")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch enterprise FastAPI REST API")
    serve_parser.add_argument("-p", "--port", type=int, default=8000, help="Port to bind FastAPI server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")

    parsed = parser.parse_args(args)

    if parsed.command == "clean":
        return run_clean_pipeline(
            input_file=parsed.dataset,
            output_file=parsed.output,
            delta_dir=parsed.delta_dir,
            report_dir=parsed.report_dir,
            auto_approve=parsed.auto_approve,
            columns=parsed.columns,
            resolve_nulls=parsed.resolve_nulls,
        )
    elif parsed.command == "ui":
        ui_script = Path(__file__).parent / "ui" / "app.py"
        print(f"[*] Launching CleanPilot UI on port {parsed.port}...")
        cmd = [sys.executable, "-m", "streamlit", "run", str(ui_script), "--server.port", str(parsed.port)]
        return subprocess.call(cmd)
    elif parsed.command == "serve":
        print(f"[*] Launching enterprise FastAPI server on http://{parsed.host}:{parsed.port}...")
        cmd = [sys.executable, "-m", "uvicorn", "narvl.api.main:app", "--host", parsed.host, "--port", str(parsed.port)]
        return subprocess.call(cmd)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

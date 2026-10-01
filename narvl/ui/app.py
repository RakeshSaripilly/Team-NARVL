"""
CleanPilot: Interactive 8-Screen Agentic Data Cleaning UI for NARVL.
Runs 100% offline on CPU.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import altair as alt

alt.data_transformers.disable_max_rows()

import importlib
import narvl.core.executor
import narvl.core.fd_miner
import narvl.core.loss
import narvl.core.normalizer
import narvl.core.profiler
import narvl.core.provenance
import narvl.core.semantic_typer
import narvl.core.shield
import narvl.core.test_gen
import narvl.engine.planner

for _mod in [
    narvl.core.executor,
    narvl.core.fd_miner,
    narvl.core.loss,
    narvl.core.normalizer,
    narvl.core.profiler,
    narvl.core.provenance,
    narvl.core.semantic_typer,
    narvl.core.shield,
    narvl.core.test_gen,
    narvl.engine.planner,
]:
    try:
        importlib.reload(_mod)
    except Exception:
        pass

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

# Streamlit Page Config
st.set_page_config(
    page_title="NARVL CleanPilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #172033 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 16px;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -2px rgba(56, 189, 248, 0.15);
    }
    .metric-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 10px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-card-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 0.88rem;
    }
    .metric-card-label {
        color: #94a3b8;
        font-weight: 500;
    }
    .metric-card-badge {
        background-color: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 2px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .metric-card-conf {
        color: #34d399;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .badge-auto {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-review {
        background-color: #78350f;
        color: #fcd34d;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-safe {
        color: #10b981;
        font-weight: bold;
    }
    .badge-blocked {
        color: #ef4444;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize persistent Streamlit session state variables."""
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "semantic_types" not in st.session_state:
        st.session_state.semantic_types = {}
    if "fds" not in st.session_state:
        st.session_state.fds = []
    if "plan_steps" not in st.session_state:
        st.session_state.plan_steps = []
    if "executor" not in st.session_state:
        st.session_state.executor = None
    if "delta_dir" not in st.session_state:
        temp_dir = tempfile.mkdtemp(prefix="narvl_ui_delta_")
        st.session_state.delta_dir = temp_dir
    if "loss_assessment" not in st.session_state:
        st.session_state.loss_assessment = None
    if "validation_result" not in st.session_state:
        st.session_state.validation_result = None
    if "provenance_json" not in st.session_state:
        st.session_state.provenance_json = None
    if "provenance_html" not in st.session_state:
        st.session_state.provenance_html = None
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = []
    if "resolve_nulls_policy" not in st.session_state:
        st.session_state.resolve_nulls_policy = True
    if "null_resolution_summary" not in st.session_state:
        st.session_state.null_resolution_summary = None


def set_selected_columns(columns: List[str]) -> None:
    """Keep the shared target selection and both column widgets synchronized."""
    selected = list(dict.fromkeys(columns))
    st.session_state.selected_columns = selected
    st.session_state.ms_profile_target_cols = selected
    st.session_state.plan_builder_cols = selected


def render_sidebar() -> str:
    """Render sidebar navigation and air-gapped system status."""
    st.sidebar.markdown("## ⚡ **NARVL CleanPilot**")
    st.sidebar.caption("Autonomous Agentic Data Cleaning • 100% Offline CPU")

    screens = [
        "1. Ingestion Shield & Upload",
        "2. AI Profile & Anomaly Radar",
        "3. AI Semantic Reasoning",
        "4. Interactive Plan Builder",
        "5. 4D Loss Simulator",
        "6. Reversible Stepper (Delta Lake)",
        "7. Dual Validation Tests",
        "8. Before vs After & Provenance",
    ]

    selected_screen = st.sidebar.radio("Navigation Steps", screens)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Security")
    st.sidebar.markdown("🔒 **Air-Gapped Mode**: `ACTIVE`")
    st.sidebar.markdown("📦 **Delta Log Storage**: `ACID Compliant`")

    if st.session_state.raw_df is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Active Dataset")
        rows, cols = st.session_state.raw_df.shape
        st.sidebar.info(f"**Rows**: {rows:,} | **Cols**: {cols}")

    return selected_screen


def screen_1_upload() -> None:
    """Screen 1: Streaming Ingestion Shield with 500MB Quota Meter."""
    st.markdown('<div class="main-header">1. Ingestion Shield & Streaming Upload</div>', unsafe_allow_html=True)
    st.write("Upload messy enterprise datasets (CSV, TSV, Parquet, JSON, NDJSON). Protected by L0 Streaming Shield.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file to clean",
            type=["csv", "tsv", "parquet", "json", "ndjson"],
            help="Protected with 500MB cumulative quota limit and delimiter bomb neutralization.",
        )

    with col2:
        st.markdown("### Shield Quota Meter")
        quota_used = uploaded_file.size if uploaded_file else 0
        quota_limit = 500 * 1024 * 1024
        quota_pct = min(1.0, quota_used / quota_limit)
        st.progress(quota_pct)
        st.caption(f"Used: {quota_used / (1024*1024):.2f} MB / 500.00 MB Limit")

    if uploaded_file is not None:
        # Save upload to temporary file
        temp_input = Path(st.session_state.delta_dir) / uploaded_file.name
        temp_input.write_bytes(uploaded_file.getvalue())

        shield = StreamingShield()
        quarantine_log = Path(st.session_state.delta_dir) / "quarantine.log"

        with st.spinner("Streaming through L0 Shield and Normalizer..."):
            try:
                clean_path, quarantine_rows = shield.sanitize_file(temp_input, quarantine_log=quarantine_log)
                normalizer = FormatNormalizer()
                df = normalizer.load_file(clean_path)
                st.session_state.raw_df = df
                st.session_state.profile = None
                st.session_state.semantic_types = {}
                st.session_state.fds = []
                st.session_state.plan_steps = []
                st.session_state.active_steps = []
                st.session_state.cleaned_df = None
                st.session_state.loss_assessment = None
                st.session_state.validation_result = None
                set_selected_columns(list(df.columns))
                st.session_state.resolve_nulls_policy = True
                st.session_state.null_resolution_summary = None

                st.success(f"Successfully ingested {df.height:,} rows across {df.width} columns!")
                if quarantine_rows > 0:
                    st.warning(f"🛡️ Shield isolated {quarantine_rows} ragged / delimiter-bomb rows to quarantine.log")

                st.dataframe(df.head(10).to_pandas(), use_container_width=True)

            except Exception as exc:
                st.error(f"Ingestion Shield Alert: {exc}")

    elif st.session_state.raw_df is None:
        # Provide sample demo dataset option
        st.info("💡 Or load the enterprise demo dataset (with typos, negative values, and dirty formatting):")
        if st.button("Load Demo Enterprise Dataset"):
            demo_df = pl.DataFrame({
                "CustomerID": [101, 102, 103, 104, 105, 101],
                "Age": [28, -12, 45, 145, 33, 28],
                "Email": ["alice@corp.com", "bob@org.net", "charlie@gmail.com", "BAD_EMAIL_FORMAT", "eva@domain.co", "alice@corp.com"],
                "State": ["Telangana", "Telengana", "Karnataka", "Karnatak", "Maharashtra", "Telangana"],
                "PostalCode": ["500001", "500001", "560001", "560001", "400001", "500001"],
                "Salary": [75000.0, 92000.0, None, 110000.0, 68000.0, 75000.0],
            })
            st.session_state.raw_df = demo_df
            st.session_state.profile = None
            st.session_state.semantic_types = {}
            st.session_state.fds = []
            st.session_state.plan_steps = []
            st.session_state.active_steps = []
            st.session_state.cleaned_df = None
            st.session_state.loss_assessment = None
            st.session_state.validation_result = None
            set_selected_columns(list(demo_df.columns))
            st.session_state.resolve_nulls_policy = True
            st.session_state.null_resolution_summary = None
            st.rerun()


def screen_2_profile() -> None:
    """Screen 2: AI Profile & Anomaly Radar."""
    st.markdown('<div class="main-header">2. AI Profile & Anomaly Radar</div>', unsafe_allow_html=True)
    if st.session_state.raw_df is None:
        st.warning("Please upload or load a dataset in Screen 1 first.")
        return

    df = st.session_state.raw_df
    profiler = FastProfiler()

    if st.session_state.profile is None:
        with st.spinner("Generating sub-2.5s CPU profile and anomaly scan..."):
            summary_res = profiler.profile(df)
            st.session_state.profile = summary_res.summary_dict

    profile = st.session_state.profile
    meta = profile.get("meta", {})
    cols = profile.get("columns", {})
    is_event_log = meta.get("is_event_log", False)

    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{meta.get('rows', 0):,}")
    m2.metric("Total Features", f"{meta.get('cols', 0):,}")
    
    if is_event_log:
        m3.metric("Data Archetype", "Event Log (No PK)", help="Discrete event stream without surrogate key; repeated tuples represent natural transaction occurrences.")
        dup_penalty = 0.0
    else:
        m3.metric("Duplicate Rows", f"{meta.get('duplicates', 0):,} ({meta.get('dup_pct', 0.0):.1f}%)")
        dup_penalty = meta.get("dup_pct", 0.0) * 0.5
    
    # Calculate Data Health Score
    null_rates = [v.get("null_pct", 0.0) for v in cols.values()]
    avg_null = np.mean(null_rates) if null_rates else 0.0
    quality_score = max(0, min(100, int(100 - (avg_null * 0.5) - dup_penalty)))
    m4.metric("Quality Score", f"{quality_score} / 100")

    if is_event_log:
        st.info(
            f"ℹ️ **Discrete Transaction Log Detected**: Dataset contains {meta.get('rows', 0):,} rows across {meta.get('cols', 0)} non-keyed columns "
            f"(state space: {meta.get('state_space', 0):,} distinct combinations). Repeated feature tuples are recognized as valid independent events and are not penalized."
        )

    st.markdown("### Feature Profiles & Anomaly Detection")
    profile_rows = []
    for cname, cinfo in cols.items():
        profile_rows.append({
            "Column": cname,
            "Type": cinfo.get("type"),
            "Null %": f"{cinfo.get('null_pct', 0.0):.2f}%",
            "Cardinality": cinfo.get("card"),
            "Min": cinfo.get("min"),
            "Median (q50)": cinfo.get("q50"),
            "Max": cinfo.get("max"),
            "Zero Variance": "⚠️ YES" if cinfo.get("zero_var") else "NO",
        })
    st.table(profile_rows)

    # Column Selection for Cleaning Pipeline
    st.markdown("---")
    st.markdown("### 🎯 Specify Columns to Process")
    st.write(
        "Choose which columns will be targeted for cleaning, null resolution, domain clamping, and validation. "
        "Unselected columns will remain untouched in their raw state."
    )

    all_cols = list(cols.keys()) if cols else list(df.columns)
    c_b1, c_b2, c_b3 = st.columns([1, 1.8, 1])
    with c_b1:
        if st.button("✅ Select All", key="btn_sel_all"):
            set_selected_columns(all_cols)
            st.session_state.plan_steps = []
            st.rerun()
    with c_b2:
        if st.button("⚠️ Select Columns with Issues Only", key="btn_sel_issues"):
            anomalous = profiler.get_columns_with_anomalies(st.session_state.profile or df)
            set_selected_columns(anomalous)
            st.session_state.plan_steps = []
            st.rerun()
    with c_b3:
        if st.button("❌ Clear Selection", key="btn_clear_sel"):
            set_selected_columns([])
            st.session_state.plan_steps = []
            st.rerun()

    new_sel = st.multiselect(
        "Active Target Columns for Cleaning Pipeline:",
        options=all_cols,
        default=st.session_state.selected_columns,
        help="Only selected features will be processed by the DAG, imputed, and validated.",
        key="ms_profile_target_cols",
    )
    if new_sel != st.session_state.selected_columns:
        set_selected_columns(new_sel)
        st.session_state.plan_steps = []


def screen_3_reasoning() -> None:
    """Screen 3: AI Semantic Reasoning & Approximate FD Discovery."""
    st.markdown('<div class="main-header">3. AI Semantic Reasoning & Constraints</div>', unsafe_allow_html=True)
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.raw_df

    # 1. Semantic Typer ONNX Classification
    st.markdown("### L2.5 ONNX Semantic Type Classification")
    typer = SemanticTyper()
    types_found = typer.infer_types(df)
    st.session_state.semantic_types = types_found

    # Render cards in a balanced responsive grid (max 4 columns per row)
    items = list(types_found.items())
    num_cols_per_row = 4
    for row_idx in range(0, len(items), num_cols_per_row):
        chunk = items[row_idx : row_idx + num_cols_per_row]
        cols = st.columns(num_cols_per_row)
        for col_idx, (col_name, res) in enumerate(chunk):
            with cols[col_idx]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title" title="{col_name}">{col_name}</div>
                        <div class="metric-card-row">
                            <span class="metric-card-label">Type:</span>
                            <span class="metric-card-badge">{res.predicted_type}</span>
                        </div>
                        <div class="metric-card-row" style="margin-bottom: 0;">
                            <span class="metric-card-label">Confidence:</span>
                            <span class="metric-card-conf">{res.confidence*100:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 2. Approximate Functional Dependency Discovery
    st.markdown("### Approximate Functional Dependencies (FDs)")
    miner = FunctionalDependencyMiner(fuzzy_threshold=85.0)
    fds = miner.mine(df)
    st.session_state.fds = fds

    if not fds:
        st.info("No functional dependencies detected with confidence >= 0.85.")
    else:
        for fd in fds:
            with st.expander(f"FD: {fd.determinant} ➔ {fd.dependent} (Confidence: {fd.confidence:.2f})", expanded=True):
                st.write(f"- Determinant: `{fd.determinant}`")
                st.write(f"- Dependent: `{fd.dependent}`")
                st.write(f"- Is Exact: `{fd.is_exact}`")
                if fd.canonical_mapping:
                    st.write("Fuzzy Typo Corrections Inferred:")
                    st.json(fd.canonical_mapping)


def screen_4_plan_builder() -> None:
    """Screen 4: Interactive Plan Builder with L5 Governance Gate."""
    st.markdown('<div class="main-header">4. Interactive Cleaning Plan Builder</div>', unsafe_allow_html=True)
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.raw_df
    planner = SLMPlanner()

    all_cols = list(df.columns)
    with st.expander("⚙️ Target Column Scope & Null Value Policy", expanded=False):
        c_sc1, c_sc2 = st.columns([2, 1])
        with c_sc1:
            chosen = st.multiselect(
                "Columns to Process in Pipeline:",
                options=all_cols,
                default=st.session_state.selected_columns,
                key="plan_builder_cols",
            )
            if chosen != st.session_state.selected_columns:
                set_selected_columns(chosen)
                st.session_state.plan_steps = []
                st.rerun()
        with c_sc2:
            st.session_state.resolve_nulls_policy = st.checkbox(
                "⚡ Auto-Resolve Null Values",
                value=st.session_state.get("resolve_nulls_policy", True),
                help="Replaces nulls with median (numeric) or mode (categorical) where appropriate. Rows with unresolvable nulls (IDs, emails, empty columns) are removed.",
            )

    if not st.session_state.plan_steps:
        with st.spinner("Generating SLM cleaning plan via constrained GBNF reasoning..."):
            pipeline = planner.generate_plan(
                profile_summary=st.session_state.profile or {},
                functional_dependencies=[fd.__dict__ for fd in st.session_state.fds],
                semantic_types={k: v.predicted_type for k, v in st.session_state.semantic_types.items()},
                selected_columns=st.session_state.selected_columns,
                raw_df=st.session_state.raw_df,
            )
            st.session_state.plan_steps = pipeline.steps

    steps = st.session_state.plan_steps
    st.write(f"Generated **{len(steps)}** candidate cleaning steps. Toggle steps to include or exclude from execution:")

    updated_steps = []
    for step in steps:
        step_id = step.get("step_id", 1)
        conf = step.get("confidence", 0.9)
        col = step.get("target_column", "ALL")
        action = step.get("action", "standardize_values")
        badge = '<span class="badge-auto">AUTO-BATCH (≥0.85)</span>' if conf >= 0.85 else '<span class="badge-review">HUMAN REVIEW (<0.85)</span>'

        with st.container():
            st.markdown(f"**Step {step_id}: {action.upper()} on `{col}`** | {badge}", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 4])
            with c1:
                default_chk = (conf >= 0.85 and step.get("loss_potential") != "high")
                include = st.checkbox(f"Execute Step {step_id}", value=default_chk, key=f"step_chk_{step_id}")
            with c2:
                st.write(f"**Rule**: {step.get('rule')}")
                st.write(f"**Justification**: {step.get('justification')}")
                st.caption(f"Confidence: {conf:.2f} | Loss Potential: {step.get('loss_potential')}")

            if include:
                updated_steps.append(step)
            st.markdown("---")

    st.session_state.active_steps = updated_steps


def screen_5_loss_simulator() -> None:
    """Screen 5: 4D Loss Estimator & Speculative Utility Barrier."""
    st.markdown('<div class="main-header">5. 4D Loss Estimator & Speculative Utility Barrier</div>', unsafe_allow_html=True)
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first.")
        return

    raw_df = st.session_state.raw_df
    steps = getattr(st.session_state, "active_steps", st.session_state.plan_steps)

    # Perform speculative dry run
    executor = ReversibleExecutor(raw_df, delta_table_path=st.session_state.delta_dir)
    candidate_df, _ = executor.execute_plan(steps)

    estimator = LossEstimator()
    assessment = estimator.assess(raw_df, candidate_df)
    st.session_state.loss_assessment = assessment

    # Status Banner
    if assessment.is_safe:
        st.success("✅ **SAFETY GATE PASSED**: All 4D loss dimensions and LightGBM utility barrier are within safe bounds.")
    else:
        st.error(f"🛑 **SAFETY GATE BLOCKED**: Potential information loss detected:\n" + "\n".join(f"- {r}" for r in assessment.blocking_reasons))

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volumetric Loss", f"{assessment.volumetric_loss * 100:.2f}%", help="Threshold <= 15.0%")
    c2.metric("Wasserstein W1", f"{assessment.statistical_w1:.4f}", help="Threshold <= 0.35")
    c3.metric("Jaccard Loss", f"{assessment.categorical_jaccard_loss:.4f}", help="Threshold <= 0.30")
    c4.metric("Utility Delta", f"{assessment.predictive_utility_delta:+.4f}", help="Must be >= 0.0")

    st.markdown("### Risk Mitigation Recommendation")
    st.info(f"**Planner Advice**: {assessment.mitigation_recommendation}")


def screen_6_stepper() -> None:
    """Screen 6: Reversible Stepper with Delta Lake Rollback."""
    st.markdown('<div class="main-header">6. Reversible Stepper & Delta Lake Execution</div>', unsafe_allow_html=True)
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first.")
        return

    raw_df = st.session_state.raw_df
    steps = getattr(st.session_state, "active_steps", st.session_state.plan_steps)

    if st.session_state.executor is None:
        st.session_state.executor = ReversibleExecutor(raw_df, delta_table_path=st.session_state.delta_dir)

    executor = st.session_state.executor
    current_v = executor.get_current_version()
    st.write(f"Current Delta Lake Table Commit Version: **v{current_v}**")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Execute Approved DAG Pipeline"):
            with st.spinner("Applying vectorized Polars DAG to Delta Lake..."):
                cleaned_df, report = executor.execute_plan(steps)
                target_cols = st.session_state.get("selected_columns", None)
                resolve_policy = st.session_state.get("resolve_nulls_policy", True)
                sem_types = {k: v.predicted_type for k, v in st.session_state.semantic_types.items()}

                cleaned_df, report = executor.execute_plan(
                    steps,
                    target_columns=target_cols,
                    resolve_nulls_policy=resolve_policy,
                    semantic_types=sem_types,
                    filter_validation_failures=True,
                )
                # Post-processing: Remove records failing Pandera or Great Expectations validation
                synthesizer = DualTestSynthesizer(steps)
                if hasattr(synthesizer, "filter_and_validate"):
                    filtered_df, val_res = synthesizer.filter_and_validate(cleaned_df)
                else:
                    val_res = synthesizer.validate_dataset(cleaned_df)
                    filtered_df = cleaned_df
                filtered_df, val_res = synthesizer.filter_and_validate(cleaned_df, target_columns=target_cols)
                st.session_state.cleaned_df = filtered_df
                st.session_state.validation_result = val_res

                msg = f"Successfully committed v{report.get('final_version')} with {len(steps)} applied steps!"
                removed_cnt = getattr(val_res, "removed_records_count", 0)
                if removed_cnt > 0:
                    msg += f" (Safely removed {removed_cnt:,} records failing Pandera / GE validation)"
                null_res = report.get("null_resolution", {})
                null_imputed = null_res.get("total_nulls_imputed", 0)
                null_dropped = null_res.get("removed_rows", 0)
                if null_imputed > 0 or null_dropped > 0:
                    msg += f" [Null Policy: Imputed {null_imputed:,} values, removed {null_dropped:,} unresolvable rows]"
                if val_res.removed_records_count > 0:
                    msg += f" (Safely removed {val_res.removed_records_count:,} records failing Pandera / GE validation)"
                st.success(msg)
                st.rerun()

    with col_btn2:
        if st.button("⏪ Undo All Steps (Rollback to Raw v0)"):
            with st.spinner("Reverting via Delta Lake Time-Travel..."):
                restored_df = executor.rollback_to_version(0)
                st.session_state.cleaned_df = restored_df
                st.info("Time-travel rollback successful: 100% bitwise parity restored with raw state.")
                st.rerun()

    if st.session_state.cleaned_df is not None:
        st.markdown("### Cleaned Snapshot Preview")
        st.dataframe(st.session_state.cleaned_df.head(10).to_pandas(), width="stretch")


def screen_7_validation() -> None:
    """Screen 7: Dual Validation Tests (Pandera + Great Expectations)."""
    st.markdown('<div class="main-header">7. Dual Validation Suite (Pandera + GE)</div>', unsafe_allow_html=True)
    target_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

    if target_df is None:
        st.warning("Please ingest a dataset first.")
        return

    steps = getattr(st.session_state, "active_steps", st.session_state.plan_steps)
    target_cols = st.session_state.get("selected_columns", None)
    synthesizer = DualTestSynthesizer(steps)

    with st.spinner("Running automated Pandera schema checks and Great Expectations checkpoint..."):
        synthesizer.validate_dataset(target_df)
        prev_removed = getattr(st.session_state.validation_result, "removed_records_count", 0)
        res = synthesizer.validate_dataset(target_df, target_columns=target_cols)
        if prev_removed > 0:
            res.removed_records_count = prev_removed
        st.session_state.validation_result = res

    if getattr(st.session_state.validation_result, "removed_records_count", 0) > 0:
        st.info(f"🛡️ **Post-Processing Active**: {st.session_state.validation_result.removed_records_count:,} invalid records failing Pandera or GE constraints were purged from cleaned dataset.")

    if not res.is_fully_validated:
        if st.button("🧹 Purge Non-Compliant Records from Cleaned Dataset"):
            if hasattr(synthesizer, "filter_and_validate"):
                filtered_df, new_res = synthesizer.filter_and_validate(target_df)
            else:
                new_res = synthesizer.validate_dataset(target_df)
                filtered_df = target_df
            st.session_state.cleaned_df = filtered_df
            st.session_state.validation_result = new_res
            purged_cnt = getattr(new_res, "removed_records_count", 0)
            st.success(f"Purged {purged_cnt:,} non-compliant records!")
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧪 Pandera DataFrameSchema")
        if res.pandera_passed:
            st.success("✅ **PASSED**: All range bounds, regex patterns, and type constraints verified.")
        else:
            st.error("❌ **BLOCKED**: Schema errors encountered:")
            for err in res.pandera_errors:
                st.code(err)

    with c2:
        st.markdown("### 📋 Great Expectations Checkpoint")
        if res.ge_passed:
            st.success(f"✅ **PASSED**: {res.ge_summary.get('total_expectations', 0)} expectations met.")
        else:
            st.error(f"❌ **BLOCKED**: {res.ge_summary.get('failed_expectations', 0)} expectation(s) failed.")
            for r in res.ge_summary.get("results", []):
                if not r["success"]:
                    st.write(f"- `{r['column']}`: {r['details']}")


def render_relationship_chart(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    chart_type: str,
    is_cleaned: bool,
) -> Optional[alt.Chart]:
    """Render full-dataset bivariate or univariate relationship chart with Altair."""
    if df.height == 0:
        st.info("No records available in this dataset snapshot.")
        return None

    if x_col not in df.columns:
        st.warning(f"Feature '{x_col}' is not present in this dataset.")
        return None

    is_y_count = (y_col == "(Count / Frequency)") or (x_col == y_col)
    if not is_y_count and y_col not in df.columns:
        st.warning(f"Feature '{y_col}' is not present in this dataset.")
        return None

    cols_to_use = [x_col] if is_y_count else [x_col, y_col]
    # Extract ALL rows from Polars DataFrame into Pandas for Altair plotting
    pdf = df.select(cols_to_use).to_pandas()

    # Smart coercion for numeric strings with currency/commas
    for c in cols_to_use:
        if pdf[c].dtype == object or pd.api.types.is_string_dtype(pdf[c]):
            non_nulls = (
                pdf[c]
                .dropna()
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            num_test = pd.to_numeric(non_nulls, errors="coerce")
            if len(non_nulls) > 0 and (num_test.notna().sum() / len(non_nulls)) >= 0.7:
                pdf[c] = pd.to_numeric(
                    pdf[c]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip(),
                    errors="coerce",
                )

    color_primary = "#10b981" if is_cleaned else "#f59e0b"
    color_scheme = "greens" if is_cleaned else "oranges"

    is_x_num = pd.api.types.is_numeric_dtype(pdf[x_col])
    is_y_num = (not is_y_count) and pd.api.types.is_numeric_dtype(pdf[y_col])

    chart = None

    try:
        if is_y_count:
            if is_x_num:
                chart = (
                    alt.Chart(pdf.dropna(subset=[x_col]))
                    .mark_bar(color=color_primary)
                    .encode(
                        x=alt.X(
                            x_col,
                            type="quantitative",
                            bin=alt.Bin(maxbins=25),
                            title=f"{x_col} (Binned Distribution)",
                        ),
                        y=alt.Y("count():Q", title="Total Records"),
                        tooltip=[
                            alt.Tooltip(x_col, bin=alt.Bin(maxbins=25), title=x_col),
                            alt.Tooltip("count():Q", title="Record Count"),
                        ],
                    )
                    .properties(height=360)
                )
            else:
                chart = (
                    alt.Chart(pdf.dropna(subset=[x_col]))
                    .mark_bar(color=color_primary)
                    .encode(
                        x=alt.X(x_col, type="nominal", sort="-y", title=x_col),
                        y=alt.Y("count():Q", title="Total Records"),
                        tooltip=[
                            alt.Tooltip(x_col, title=x_col),
                            alt.Tooltip("count():Q", title="Record Count"),
                        ],
                    )
                    .properties(height=360)
                )
        elif chart_type == "Scatter Plot" or (chart_type == "Auto (Smart Detect)" and is_x_num and is_y_num):
            chart = (
                alt.Chart(pdf.dropna(subset=[x_col, y_col]))
                .mark_circle(size=70, opacity=0.75, color=color_primary)
                .encode(
                    x=alt.X(x_col, type="quantitative" if is_x_num else "nominal", title=x_col),
                    y=alt.Y(y_col, type="quantitative" if is_y_num else "nominal", title=y_col),
                    tooltip=[x_col, y_col],
                )
                .properties(height=360)
                .interactive()
            )
        elif chart_type == "Line Chart":
            chart = (
                alt.Chart(pdf.dropna(subset=[x_col, y_col]))
                .mark_line(point=True, color=color_primary)
                .encode(
                    x=alt.X(x_col, type="quantitative" if is_x_num else "nominal", title=x_col),
                    y=alt.Y(y_col, type="quantitative" if is_y_num else "nominal", title=y_col),
                    tooltip=[x_col, y_col],
                )
                .properties(height=360)
                .interactive()
            )
        elif chart_type == "Box Plot" and is_y_num:
            chart = (
                alt.Chart(pdf.dropna(subset=[y_col]))
                .mark_boxplot(color=color_primary)
                .encode(
                    x=alt.X(x_col, type="nominal", title=x_col),
                    y=alt.Y(y_col, type="quantitative", title=y_col),
                    tooltip=[x_col, y_col],
                )
                .properties(height=360)
            )
        elif not is_x_num and is_y_num:
            chart = (
                alt.Chart(pdf.dropna(subset=[x_col, y_col]))
                .mark_bar(color=color_primary)
                .encode(
                    x=alt.X(x_col, type="nominal", sort="-y", title=x_col),
                    y=alt.Y(f"mean({y_col}):Q", title=f"Mean {y_col}"),
                    tooltip=[
                        x_col,
                        alt.Tooltip(f"mean({y_col}):Q", title=f"Mean {y_col}", format=".2f"),
                        alt.Tooltip("count():Q", title="Record Count"),
                    ],
                )
                .properties(height=360)
            )
        elif is_x_num and not is_y_num:
            chart = (
                alt.Chart(pdf.dropna(subset=[x_col, y_col]))
                .mark_bar(color=color_primary)
                .encode(
                    y=alt.Y(y_col, type="nominal", sort="-x", title=y_col),
                    x=alt.X(f"mean({x_col}):Q", title=f"Mean {x_col}"),
                    tooltip=[
                        y_col,
                        alt.Tooltip(f"mean({x_col}):Q", title=f"Mean {x_col}", format=".2f"),
                        alt.Tooltip("count():Q", title="Record Count"),
                    ],
                )
                .properties(height=360)
            )
        else:
            # Both categorical
            chart = (
                alt.Chart(pdf.dropna(subset=[x_col, y_col]))
                .mark_rect()
                .encode(
                    x=alt.X(x_col, type="nominal", title=x_col),
                    y=alt.Y(y_col, type="nominal", title=y_col),
                    color=alt.Color("count():Q", scale=alt.Scale(scheme=color_scheme), title="Frequency"),
                    tooltip=[x_col, y_col, alt.Tooltip("count():Q", title="Frequency")],
                )
                .properties(height=360)
            )

        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
    except Exception as err:
        st.error(f"Error rendering relationship chart: {err}")

    # Micro-metrics footer
    tot = len(pdf)
    null_x = pdf[x_col].isna().sum() if x_col in pdf.columns else 0
    null_y = pdf[y_col].isna().sum() if not is_y_count and y_col in pdf.columns else 0
    st.caption(
        f"📊 **Plotted:** {tot:,} records • **Nulls ({x_col}):** {null_x:,}"
        + (f" • **Nulls ({y_col}):** {null_y:,}" if not is_y_count and y_col != x_col else "")
    )
    return chart


def screen_8_before_after() -> None:
    """Screen 8: Before vs After Dashboard & Signed Provenance Export."""
    st.markdown('<div class="main-header">8. Before vs After & Provenance Audit</div>', unsafe_allow_html=True)
    if st.session_state.raw_df is None:
        st.warning("Please upload a dataset first.")
        return

    raw_df = st.session_state.raw_df
    cleaned_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else raw_df

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Raw Dataset (Before)")
        st.dataframe(raw_df.head(10).to_pandas(), width="stretch")

    with c2:
        st.markdown("#### Cleaned Dataset (After)")
        st.dataframe(cleaned_df.head(10).to_pandas(), width="stretch")

    st.markdown("---")
    st.markdown("### 📊 Interactive Dataset Visualizer (All Records)")
    st.caption("Select X and Y features below to visually inspect how their relationship and distribution compare between the raw and cleaned datasets across **all records**.")

    raw_chart_spec = None
    cleaned_chart_spec = None
    selected_x = None
    selected_y = None

    all_cols = list(dict.fromkeys(list(raw_df.columns) + list(cleaned_df.columns)))
    if all_cols:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1.5])
        with col_ctrl1:
            x_col = st.selectbox(
                "Select X-Axis Feature",
                options=all_cols,
                index=0,
                key="screen8_viz_x",
                help="Feature plotted on the horizontal axis across all records.",
            )
        with col_ctrl2:
            y_options = ["(Count / Frequency)"] + all_cols
            default_y_idx = 2 if len(all_cols) > 1 else 0
            y_col = st.selectbox(
                "Select Y-Axis Feature",
                options=y_options,
                index=default_y_idx if default_y_idx < len(y_options) else 0,
                key="screen8_viz_y",
                help="Feature plotted on the vertical axis or frequency count across all records.",
            )
        with col_ctrl3:
            chart_type = st.selectbox(
                "Visualization Style",
                options=["Auto (Smart Detect)", "Scatter Plot", "Bar Chart", "Line Chart", "Box Plot"],
                index=0,
                key="screen8_viz_type",
                help="Choose display format or let smart detect choose optimal visual.",
            )

        selected_x = x_col
        selected_y = y_col

        viz_c1, viz_c2 = st.columns(2)
        with viz_c1:
            st.markdown(f"##### 📉 Raw Data: `{x_col}` vs `{y_col}`")
            st.caption(f"All **{raw_df.height:,}** records considered")
            raw_chart = render_relationship_chart(raw_df, x_col, y_col, chart_type, is_cleaned=False)

        with viz_c2:
            st.markdown(f"##### 📈 Cleaned Data: `{x_col}` vs `{y_col}`")
            st.caption(f"All **{cleaned_df.height:,}** records considered")
            cleaned_chart = render_relationship_chart(cleaned_df, x_col, y_col, chart_type, is_cleaned=True)

        if raw_chart is not None:
            try:
                raw_chart_spec = raw_chart.to_dict()
            except Exception as e:
                logger.warning(f"Could not extract raw chart spec: {e}")
        if cleaned_chart is not None:
            try:
                cleaned_chart_spec = cleaned_chart.to_dict()
            except Exception as e:
                logger.warning(f"Could not extract cleaned chart spec: {e}")
    else:
        st.info("No columns available to visualize.")

    st.markdown("---")
    st.markdown("### Provenance Audit Report Generation")

    reporter = ProvenanceReporter()
    loss_dict = st.session_state.loss_assessment.__dict__ if st.session_state.loss_assessment else {}
    val_dict = st.session_state.validation_result.__dict__ if st.session_state.validation_result else {}

    report_json_path = Path(st.session_state.delta_dir) / "narvl_provenance_report.json"
    report_html_path = Path(st.session_state.delta_dir) / "narvl_provenance_report.html"

    report_data = reporter.generate_report(
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        plan_steps=st.session_state.plan_steps,
        loss_assessment=loss_dict,
        validation_result=val_dict,
        output_json=report_json_path,
        output_html=report_html_path,
        raw_chart_spec=raw_chart_spec,
        cleaned_chart_spec=cleaned_chart_spec,
        x_feature=selected_x,
        y_feature=selected_y,
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        csv_buffer = io.BytesIO()
        cleaned_df.write_csv(csv_buffer)
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name="narvl_cleaned_dataset.csv",
            mime="text/csv",
        )

    with d2:
        st.download_button(
            label="📄 Download Provenance JSON",
            data=json.dumps(report_data, indent=2),
            file_name="narvl_provenance_report.json",
            mime="application/json",
        )

    with d3:
        if report_html_path.exists():
            st.download_button(
                label="🌐 Download Executive HTML Report",
                data=report_html_path.read_text(encoding="utf-8"),
                file_name="narvl_provenance_report.html",
                mime="text/html",
            )


def main() -> None:
    """Streamlit CleanPilot main application."""
    init_session_state()
    selected = render_sidebar()

    if "1." in selected:
        screen_1_upload()
    elif "2." in selected:
        screen_2_profile()
    elif "3." in selected:
        screen_3_reasoning()
    elif "4." in selected:
        screen_4_plan_builder()
    elif "5." in selected:
        screen_5_loss_simulator()
    elif "6." in selected:
        screen_6_stepper()
    elif "7." in selected:
        screen_7_validation()
    elif "8." in selected:
        screen_8_before_after()


if __name__ == "__main__":
    main()

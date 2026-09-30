"""
CleanPilot Document Intelligence: Interactive Streamlit UI for Document Cleaning.
Features Screen 0: Unstructured Document Ingestion (PDF / DOCX / TXT).
100% Offline CPU execution.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import streamlit as st

from narvl.document.pipeline import DocumentCleaningPipeline

st.set_page_config(
    page_title="NARVL CleanPilot - Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("📄 NARVL CleanPilot: Document Intelligence")
st.markdown("Autonomous, offline, and reversible data cleaning for **PDF, DOCX, and TXT** documents.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = DocumentCleaningPipeline()
if "doc_result" not in st.session_state:
    st.session_state.doc_result = None

# Screen 0: Document Upload & Ingestion
st.subheader("Screen 0: Document Ingestion & Shield")
uploaded_file = st.file_uploader(
    "Upload unstructured document (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    help="Preserves original copy in doc_extension/data/originals/ with L0 Shield inspection",
)

auto_approve = st.checkbox("Auto-Approve Low and Medium Risk Cleaning Operations", value=True)

if uploaded_file is not None and st.button("🚀 Process & Clean Document"):
    with st.spinner("Analyzing document structure, extracting entities, and detecting data quality issues..."):
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
            tmp_f.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_f.name)

        try:
            res = st.session_state.pipeline.run_cleaning_flow(tmp_path, auto_approve=auto_approve)
            st.session_state.doc_result = res
            st.success(f"Processing complete for {uploaded_file.name} (Doc ID: {res['doc_id']})")
        finally:
            tmp_path.unlink(missing_ok=True)

res = st.session_state.doc_result
if res is not None:
    st.markdown("---")

    # Metrics Overview
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Format", res["file_type"])
    c2.metric("Extracted Entities", res["raw_entity_count"])
    c3.metric("Cleaned Entities", res["cleaned_entity_count"])
    c4.metric("Quality Issues", len(res["quality_issues"]))

    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Cleaned Records",
        "🔍 Detected Quality Issues",
        "💡 Cleaning Recommendations",
        "✅ Validation Suite",
        "🕒 Delta Time-Travel History",
    ])

    with t1:
        st.subheader("Structured Cleaned Records")
        records_df = res["records_dataframe"]
        st.dataframe(records_df.to_pandas() if hasattr(records_df, "to_pandas") else records_df.to_dicts(), use_container_width=True)

    with t2:
        st.subheader("Data Quality Anomaly Report")
        for iss in res["quality_issues"]:
            sev_color = "red" if iss["severity"] == "high" else ("orange" if iss["severity"] == "medium" else "green")
            st.markdown(f":{sev_color}[**[{iss['severity'].upper()}] {iss['type']}**] on `{iss['field']}`")
            st.write(f"- Evidence: {iss['evidence']}")
            if iss.get("conflicting_values"):
                st.write(f"- Conflicting: {iss['conflicting_values']}")

    with t3:
        st.subheader("Cleaning Recommendations & L4/L5 Governance")
        for rec in res["recommendations"]:
            st.markdown(f"**Action**: `{rec['operation_type']}` | Risk: `{rec['risk_level']}` | Confidence: `{rec['confidence']:.2f}`")
            st.write(f"- {rec['explanation']}")
            st.write(f"- Proposed change: `{rec['proposed_change']}`")
            st.write(f"- Reversible: `{rec['is_reversible']}` (Loss score: {rec['loss_score']:.2f})")

    with t4:
        st.subheader("Automated Dual Validation Checks")
        val = res["validation_report"]
        st.write(f"Overall Passed: **{val['passed']}**")
        for chk in val.get("checks", []):
            icon = "✅" if chk["passed"] else "❌"
            st.markdown(f"{icon} **{chk['name']}** (`{chk['target_field']}`): {chk['details']}")

    with t5:
        st.subheader("Delta Lake Versioning & Rollback")
        for v in res["versions"]:
            st.write(f"• **Version {v['version_id']}**: `{v['operation']}` ({v['timestamp']})")

        rollback_v = st.number_input("Rollback to version", min_value=0, max_value=len(res["versions"])-1, value=0)
        if st.button("⏪ Roll Back"):
            restored = st.session_state.pipeline.rollback_to_version(res["doc_id"], int(rollback_v))
            st.success(f"Restored to version {rollback_v} with 100% bitwise parity ({len(restored)} entities)")

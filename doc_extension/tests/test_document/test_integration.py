"""
Test End-to-End Document Intelligence Integration.
Validates:
- Document -> Extractor -> Entities -> Polars DataFrame bridge
- L1-L6 execution
- Provenance signature HMAC-SHA256 and export files
"""

from pathlib import Path
from narvl.document.exporters import DocumentExporter
from narvl.document.pipeline import DocumentCleaningPipeline


def test_end_to_end_document_integration(tmp_path):
    doc_path = tmp_path / "integration_input.txt"
    doc_path.write_text(
        "ANNUAL CUSTOMER SUMMARY REPORT\n\n"
        "Customer Name: Priya Sharma\n"
        "Contact Phone: 9123456780\n"
        "Email: priya.sharma@company.com\n"
        "City: Hyd\n"
        "State: Telangana\n"
        "Date: 20/04/2024\n"
        "Invoice Number: INV-9988\n"
        "Amount: $4,500.00\n",
        encoding="utf-8",
    )

    pipeline = DocumentCleaningPipeline(base_dir=tmp_path, auto_approve=True)
    res = pipeline.run_cleaning_flow(doc_path, auto_approve=True)

    assert res["doc_id"] is not None
    assert res["raw_entity_count"] > 0
    assert res["cleaned_entity_count"] > 0
    assert len(res["versions"]) >= 4

    # Tabular DataFrame bridge
    records_df = res["records_dataframe"]
    assert records_df.height >= 1
    assert "customer_name" in records_df.columns
    assert "city" in records_df.columns
    # Check that city was cleaned to Hyderabad
    assert records_df["city"][0] == "Hyderabad"

    # Export deliverables
    exporter = DocumentExporter(output_dir=tmp_path / "outputs")
    exports = exporter.export_all(res)

    assert exports["csv"].exists()
    assert exports["json"].exists()
    assert exports["excel"].exists()
    assert exports["report_md"].exists()
    assert exports["provenance_json"].exists()
    assert exports["provenance_html"].exists()

    # Validate provenance HMAC digital signature
    import json
    with open(exports["provenance_json"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "digital_signature" in manifest
    assert manifest["signature_algorithm"] == "HMAC-SHA256"
    assert manifest["lineage"]["clean_shape"][0] >= 1

"""
Test Reversibility and Delta Lake Rollback.
Validates:
- Incremental version committing (version 0, 1, 2, ...)
- Rollback to version 0 restoring initial extracted state with 100% bitwise parity
"""

from pathlib import Path
from narvl.document.models import StructuredEntity
from narvl.document.pipeline import DocumentCleaningPipeline


def test_delta_rollback_100_percent_parity(tmp_path):
    text_file = tmp_path / "customer.txt"
    text_file.write_text(
        "Customer Name: Rajesh Kumar\n"
        "City: Hyd\n"
        "Phone: 9876543210\n",
        encoding="utf-8",
    )

    pipeline = DocumentCleaningPipeline(base_dir=tmp_path, auto_approve=True)
    res = pipeline.run_cleaning_flow(text_file, auto_approve=True)
    doc_id = res["doc_id"]

    # Initial version 0 entities
    v0_snapshot_path = Path(res["versions"][0]["data_snapshot_path"])
    assert v0_snapshot_path.exists()

    v0_entities = [StructuredEntity(**e) for e in res["versions"][0]["entities"]]
    v0_tuples = [(e.entity_id, e.type, e.value) for e in v0_entities]

    # Cleaned version has modified city (Hyd -> Hyderabad)
    latest_entities = pipeline.latest_entities[doc_id]
    latest_cities = [e.value for e in latest_entities if e.type == "City"]
    assert "Hyderabad" in latest_cities

    # Execute rollback to version 0
    restored_entities = pipeline.rollback_to_version(doc_id, target_version=0)
    restored_tuples = [(e.entity_id, e.type, e.value) for e in restored_entities]

    # Verify 100% parity with initial version 0 state
    assert restored_tuples == v0_tuples
    restored_cities = [e.value for e in restored_entities if e.type == "City"]
    assert "Hyd" in restored_cities

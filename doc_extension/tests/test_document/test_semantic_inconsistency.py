"""
Test Semantic Inconsistency and RapidFuzz Canonicalization.
Validates:
- Hyd, HYD, Hyderabad -> Hyderabad canonicalization
- Telengana -> Telangana 100% confidence typo repair
"""

from narvl.document.entity_extractor import EntityExtractor
from narvl.document.models import DocumentType, Page, ParsedDocument, StructuredEntity
from narvl.document.pipeline import DocumentCleaningPipeline
from narvl.document.quality_analyzer import QualityAnalyzer


def test_city_and_state_semantic_clustering(tmp_path):
    text = (
        "Customer Name: Ramesh Kumar\n"
        "City: Hyd\n"
        "State: Telengana\n\n"
        "Customer Name: Suresh Babu\n"
        "City: HYD\n"
        "State: Telangana\n\n"
        "Customer Name: Rajesh Rao\n"
        "City: Hyderabad\n"
        "State: Telangana\n"
    )
    doc_file = tmp_path / "semantic_doc.txt"
    doc_file.write_text(text, encoding="utf-8")

    pipeline = DocumentCleaningPipeline(base_dir=tmp_path, auto_approve=True)
    res = pipeline.run_cleaning_flow(doc_file, auto_approve=True)

    # All city variants must be cleaned to Hyderabad
    cleaned_entities = [StructuredEntity(**e) for e in res["versions"][-1]["entities"]]
    cities = [e.value for e in cleaned_entities if e.type == "City"]
    assert len(cities) >= 1
    assert all(c == "Hyderabad" for c in cities), f"Expected all cities to be 'Hyderabad', got: {cities}"

    # All state variants must be cleaned to Telangana
    states = [e.value for e in cleaned_entities if e.type == "State"]
    assert len(states) >= 1
    assert all(s == "Telangana" for s in states), f"Expected all states to be 'Telangana', got: {states}"

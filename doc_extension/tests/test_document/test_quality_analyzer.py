"""
Test Quality Analyzer across all 6 issue types:
1. missing
2. duplicate
3. format_inconsistent
4. conflict
5. invalid
6. semantic
"""

from narvl.document.models import StructuredEntity
from narvl.document.quality_analyzer import QualityAnalyzer
from narvl.document.relation_detector import EntityGraph, RelationDetector


def test_quality_analyzer_six_issue_types():
    analyzer = QualityAnalyzer()

    entities = [
        # Duplicate / Near-duplicate
        StructuredEntity(entity_id="e1", type="CustomerName", value="Rajesh Kumar", source_page=1),
        StructuredEntity(entity_id="e2", type="CustomerName", value="Rajesh Kumar", source_page=2),
        # Inconsistent format
        StructuredEntity(entity_id="e3", type="Phone", value="9876543210", normalized_value="+91-98765-43210", source_page=1),
        StructuredEntity(entity_id="e4", type="Phone", value="+91-98765-43210", normalized_value="+91-98765-43210", source_page=2),
        # Inconsistent Date
        StructuredEntity(entity_id="e5", type="Date", value="15/03/2024", normalized_value="2024-03-15", source_page=1),
        # Semantic variant
        StructuredEntity(entity_id="e6", type="City", value="Hyd", normalized_value="Hyderabad", source_page=1),
        StructuredEntity(entity_id="e7", type="City", value="Hyderabad", normalized_value="Hyderabad", source_page=2),
        StructuredEntity(entity_id="e8", type="State", value="Telengana", normalized_value="Telangana", source_page=1),
        # Invalid
        StructuredEntity(entity_id="e9", type="Email", value="not-an-email", source_page=1),
        StructuredEntity(entity_id="e10", type="Phone", value="12345", source_page=1),
    ]

    # Graph with missing email and conflicting phone
    detector = RelationDetector()
    graph = detector.detect_relations(entities)

    issues = analyzer.analyze(entities, graph)
    issue_types = {iss.type for iss in issues}

    assert "duplicate" in issue_types, f"Expected 'duplicate' in {issue_types}"
    assert "format_inconsistent" in issue_types, f"Expected 'format_inconsistent' in {issue_types}"
    assert "semantic" in issue_types, f"Expected 'semantic' in {issue_types}"
    assert "invalid" in issue_types, f"Expected 'invalid' in {issue_types}"
    assert "missing" in issue_types, f"Expected 'missing' in {issue_types}"

    # Verify conflict detection
    conflict_graph = EntityGraph(
        clusters=[
            {
                "primary_entity": "Same Customer",
                "type": "CustomerName",
                "attributes": {
                    "City": ["Hyderabad", "New York"],  # Conflicting cities
                },
            }
        ]
    )
    conflict_issues = analyzer.analyze([], conflict_graph)
    assert any(iss.type == "conflict" for iss in conflict_issues)

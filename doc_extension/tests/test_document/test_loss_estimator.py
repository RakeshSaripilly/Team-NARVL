"""
Test Document Loss Estimator.
Validates:
- Low-risk reversible transformations: date (0.1), phone (0.2), city (0.15)
- Medium-risk operations requiring approval: fuzzy deduplication (0.5)
- High-risk blocking operations: conflicting entity merge (0.9, BLOCKED)
"""

from narvl.document.cleaning_planner import DocumentCleaningPlanner
from narvl.document.loss_estimator import DocumentLossEstimator
from narvl.document.models import QualityIssue


def test_loss_estimator_risk_stratification():
    estimator = DocumentLossEstimator()

    # 1. Date normalization (low risk 0.1, reversible)
    iss_date = QualityIssue(
        issue_id="iss_1",
        type="format_inconsistent",
        field="Date",
        current_value="15/03/2024",
        conflicting_values=["2024-03-15"],
    )
    score, reason, risk, rev, blocked = estimator.evaluate_recommendation_loss(
        iss_date, {"from": "15/03/2024", "to": "2024-03-15"}, "standardize"
    )
    assert score == 0.10
    assert risk == "low"
    assert rev is True
    assert blocked is False

    # 2. Phone E.164 (low risk 0.2, reversible)
    iss_phone = QualityIssue(
        issue_id="iss_2",
        type="format_inconsistent",
        field="Phone",
        current_value="9876543210",
        conflicting_values=["+91-98765-43210"],
    )
    score, reason, risk, rev, blocked = estimator.evaluate_recommendation_loss(
        iss_phone, {"from": "9876543210", "to": "+91-98765-43210"}, "standardize"
    )
    assert score == 0.20
    assert risk == "low"
    assert rev is True
    assert blocked is False

    # 3. City Hyd -> Hyderabad (low risk 0.15)
    iss_city = QualityIssue(
        issue_id="iss_3",
        type="semantic",
        field="City",
        current_value="Hyd",
        conflicting_values=["Hyderabad"],
    )
    score, reason, risk, rev, blocked = estimator.evaluate_recommendation_loss(
        iss_city, {"from": "Hyd", "to": "Hyderabad"}, "standardize"
    )
    assert score == 0.15
    assert risk == "low"
    assert rev is True
    assert blocked is False

    # 4. Fuzzy deduplication (medium risk 0.5)
    iss_dup = QualityIssue(
        issue_id="iss_4",
        type="duplicate",
        field="CustomerName",
        current_value="Rajesh Kumar",
        conflicting_values=["Rajesh Kumar"],
    )
    score, reason, risk, rev, blocked = estimator.evaluate_recommendation_loss(
        iss_dup, {"duplicate_record": "Rajesh Kumar"}, "deduplicate"
    )
    assert score == 0.50
    assert risk == "medium"
    assert rev is True
    assert blocked is False

    # 5. Conflicting entity merge (high risk 0.9, BLOCKED)
    iss_confl = QualityIssue(
        issue_id="iss_5",
        type="conflict",
        field="Phone",
        current_value="9876543210",
        conflicting_values=["9123456789"],
    )
    score, reason, risk, rev, blocked = estimator.evaluate_recommendation_loss(
        iss_confl, {"from": "9876543210", "to": "9123456789"}, "merge"
    )
    assert score == 0.90
    assert risk == "high"
    assert rev is False
    assert blocked is True
    assert "BLOCK" in reason


def test_planner_governance_routing():
    planner = DocumentCleaningPlanner()
    issues = [
        QualityIssue(issue_id="i1", type="semantic", field="City", current_value="Hyd", conflicting_values=["Hyderabad"]),
        QualityIssue(issue_id="i2", type="conflict", field="Phone", current_value="111", conflicting_values=["222"]),
    ]
    recs = planner.plan(issues)

    rec_map = {r.issue_id: r for r in recs}
    assert rec_map["i1"].requires_approval is False  # low risk + high confidence -> auto
    assert rec_map["i2"].requires_approval is True   # high risk / blocked -> requires approval

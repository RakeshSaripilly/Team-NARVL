"""
Document Cleaning Planner synthesizing actionable CleaningRecommendations from QualityIssues.
Integrates with L4 Loss Estimator and L5 Confidence Governance Gate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from narvl.document.loss_estimator import DocumentLossEstimator
from narvl.document.models import (
    CleaningRecommendation,
    OperationType,
    QualityIssue,
    StructuredEntity,
)

logger = logging.getLogger("narvl.document.cleaning_planner")


class DocumentCleaningPlanner:
    """Plans document cleaning operations and routes through L5 Governance Gate."""

    def __init__(
        self,
        loss_estimator: Optional[DocumentLossEstimator] = None,
        confidence_threshold: float = 0.85,
    ) -> None:
        self.loss_estimator = loss_estimator or DocumentLossEstimator()
        self.confidence_threshold = confidence_threshold

    def plan(
        self,
        issues: List[QualityIssue],
        entities: Optional[List[StructuredEntity]] = None,
    ) -> List[CleaningRecommendation]:
        """Generate CleaningRecommendation for each detected QualityIssue."""
        recommendations: List[CleaningRecommendation] = []

        for issue in issues:
            op_type: OperationType = "normalize"
            proposed_change: Dict[str, Any] = {}
            explanation = ""
            confidence = 0.90

            if issue.type == "semantic":
                # E.g. Hyd, HYD -> Hyderabad, Telengana -> Telangana
                op_type = "standardize"
                to_val = issue.conflicting_values[0] if issue.conflicting_values else ""
                proposed_change = {"from": issue.current_value, "to": to_val}
                explanation = f"Canonicalize semantic variant '{issue.current_value}' to standard '{to_val}'"
                confidence = 0.98

            elif issue.type == "format_inconsistent":
                op_type = "standardize"
                to_val = issue.conflicting_values[0] if issue.conflicting_values else ""
                proposed_change = {"from": issue.current_value, "to": to_val}
                explanation = f"Standardize formatting for {issue.field}: '{issue.current_value}' -> '{to_val}'"
                confidence = 0.95

            elif issue.type == "duplicate":
                op_type = "deduplicate"
                to_val = issue.conflicting_values[0] if issue.conflicting_values else ""
                proposed_change = {"duplicate_record": issue.current_value, "keep_canonical": to_val}
                explanation = f"Deduplicate redundant {issue.field} record '{issue.current_value}'"
                confidence = 0.88

            elif issue.type == "conflict":
                op_type = "merge"
                to_val = issue.conflicting_values[0] if issue.conflicting_values else ""
                proposed_change = {"from": issue.current_value, "to": to_val}
                explanation = f"Conflicting {issue.field} values across sources: '{issue.current_value}' vs '{to_val}'"
                confidence = 0.60

            elif issue.type == "missing":
                op_type = "fill"
                proposed_change = {"field": issue.field, "action": "impute_or_flag"}
                explanation = f"Missing required field {issue.field}"
                confidence = 0.70

            elif issue.type == "invalid":
                op_type = "normalize"
                proposed_change = {"invalid_value": issue.current_value, "action": "quarantine_or_correct"}
                explanation = f"Malformed/invalid {issue.field}: '{issue.current_value}'"
                confidence = 0.80

            # Evaluate with L4 Loss Estimator
            loss_score, loss_reason, risk_level, is_reversible, is_blocked = (
                self.loss_estimator.evaluate_recommendation_loss(
                    issue=issue,
                    proposed_change=proposed_change,
                    operation_type=op_type,
                )
            )

            # L5 Governance Gate:
            # Route to auto_batch if confidence >= threshold AND risk is low AND not blocked
            requires_approval = (
                confidence < self.confidence_threshold
                or risk_level in ["medium", "high"]
                or is_blocked
            )

            rec = CleaningRecommendation(
                issue_id=issue.issue_id,
                explanation=explanation,
                proposed_change=proposed_change,
                operation_type=op_type,
                confidence=confidence,
                risk_level=risk_level,  # type: ignore
                loss_score=loss_score,
                loss_reason=loss_reason,
                is_reversible=is_reversible,
                requires_approval=requires_approval,
            )
            recommendations.append(rec)

        return recommendations

"""
Approval Manager for L5 Confidence Governance Gate.
Enables fine-grained user inspection, approve/reject workflow, and audit logging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from narvl.document.models import CleaningRecommendation


class ApprovalManager:
    """Manages the review queue and user approvals for cleaning recommendations."""

    def __init__(self, recommendations: Optional[List[CleaningRecommendation]] = None) -> None:
        self.recommendations: Dict[str, CleaningRecommendation] = {}
        self.approved_ids: set[str] = set()
        self.rejected_ids: Dict[str, str] = {}  # issue_id -> reason

        if recommendations:
            self.load_recommendations(recommendations)

    def load_recommendations(self, recs: List[CleaningRecommendation]) -> None:
        """Load recommendations and automatically approve low-risk items."""
        for r in recs:
            self.recommendations[r.issue_id] = r
            if not r.requires_approval:
                self.approved_ids.add(r.issue_id)

    def approve(self, issue_id: str) -> bool:
        """Approve a recommendation by issue_id."""
        if issue_id in self.recommendations:
            self.approved_ids.add(issue_id)
            self.rejected_ids.pop(issue_id, None)
            return True
        return False

    def reject(self, issue_id: str, reason: str = "Rejected by user") -> bool:
        """Reject a recommendation by issue_id."""
        if issue_id in self.recommendations:
            self.rejected_ids[issue_id] = reason
            self.approved_ids.discard(issue_id)
            return True
        return False

    def auto_approve_all(self, include_high_risk: bool = False) -> int:
        """Batch approve recommendations based on risk criteria."""
        count = 0
        for issue_id, r in self.recommendations.items():
            if issue_id in self.rejected_ids:
                continue
            if not include_high_risk and r.risk_level == "high":
                # High risk items remain blocked unless explicitly allowed
                continue
            if issue_id not in self.approved_ids:
                self.approved_ids.add(issue_id)
                count += 1
        return count

    def get_approved(self) -> List[CleaningRecommendation]:
        """Return list of all approved recommendations."""
        return [self.recommendations[iid] for iid in self.approved_ids if iid in self.recommendations]

    def get_pending(self) -> List[CleaningRecommendation]:
        """Return list of recommendations still requiring human review."""
        return [
            r
            for iid, r in self.recommendations.items()
            if iid not in self.approved_ids and iid not in self.rejected_ids
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Return audit summary of the governance gate state."""
        return {
            "total_recommendations": len(self.recommendations),
            "approved_count": len(self.approved_ids),
            "rejected_count": len(self.rejected_ids),
            "pending_count": len(self.get_pending()),
            "auto_approved": sum(1 for r in self.recommendations.values() if not r.requires_approval),
            "human_review_required": sum(1 for r in self.recommendations.values() if r.requires_approval),
        }

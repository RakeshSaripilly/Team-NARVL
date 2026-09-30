"""
Document Loss Estimator extending L4 4D Information Loss and Speculative Utility Barrier.
Provides:
- 4D metric calculations (Volumetric, W1, Jaccard, Cosine Drift, Predictive Utility)
- Document-specific information loss scoring:
  * date normalization: 0.10 (low risk, reversible)
  * phone E.164 standardization: 0.20 (low risk, reversible)
  * Hyd/HYD/Hyderabad canonicalization: 0.15 (low risk, reversible)
  * fuzzy deduplication (95%+): 0.50 (medium risk, requires approval)
  * merging conflicting entities (same name, different phones): 0.90 (high risk, BLOCKED)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from narvl.core.loss import (
    COSINE_DRIFT_THRESHOLD,
    LossEstimator,
    LossImpactReport,
    VOLUMETRIC_THRESHOLD,
    WASSERSTEIN_THRESHOLD,
)
from narvl.document.models import CleaningRecommendation, QualityIssue

logger = logging.getLogger("narvl.document.loss_estimator")


class DocumentLossEstimator:
    """Estimates information loss for document-level transformations and tabular conversions."""

    def __init__(
        self,
        volumetric_threshold: float = VOLUMETRIC_THRESHOLD,
        wasserstein_threshold: float = WASSERSTEIN_THRESHOLD,
        cosine_threshold: float = COSINE_DRIFT_THRESHOLD,
    ) -> None:
        self.tabular_estimator = LossEstimator(
            volumetric_threshold=volumetric_threshold,
            wasserstein_threshold=wasserstein_threshold,
            cosine_threshold=cosine_threshold,
        )

    def evaluate_recommendation_loss(
        self,
        issue: QualityIssue,
        proposed_change: Dict[str, Any],
        operation_type: str,
    ) -> Tuple[float, str, str, bool, bool]:
        """Evaluate information loss for a single document cleaning recommendation.
        
        Returns:
            Tuple of (loss_score, loss_reason, risk_level, is_reversible, is_blocked)
        """
        from_val = str(proposed_change.get("from", ""))
        to_val = str(proposed_change.get("to", ""))

        # 1. Conflict / High-Risk Merges
        if issue.type == "conflict":
            return (
                0.90,
                f"BLOCK: Merging conflicting {issue.field} values ('{from_val}' vs '{to_val}') violates identity integrity barrier.",
                "high",
                False,
                True,  # is_blocked = True
            )

        # 2. Fuzzy Deduplication
        if operation_type == "deduplicate":
            return (
                0.50,
                "Entity deduplication eliminates redundant records; requires human sign-off.",
                "medium",
                True,
                False,
            )

        # 3. Phone E.164 standardization
        if issue.field == "Phone" and operation_type == "standardize":
            return (
                0.20,
                f"Standardized phone '{from_val}' to E.164 format '{to_val}'. Fully reversible.",
                "low",
                True,
                False,
            )

        # 4. Semantic City / State Canonicalization (Hyd / HYD -> Hyderabad)
        if issue.field in ["City", "State"] and operation_type in ["standardize", "normalize"]:
            return (
                0.15,
                f"Mapped variant '{from_val}' to canonical root '{to_val}'. Preserves semantic entropy.",
                "low",
                True,
                False,
            )

        # 5. Date normalization (e.g. DD/MM/YYYY -> ISO-8601)
        if issue.field == "Date" and operation_type in ["standardize", "normalize"]:
            return (
                0.10,
                f"Normalized date '{from_val}' to ISO 8601 standard '{to_val}'. Fully reversible.",
                "low",
                True,
                False,
            )

        # 6. Fill / Imputation
        if operation_type == "fill":
            return (
                0.35,
                f"Imputed missing field '{issue.field}' with '{to_val}'.",
                "medium",
                True,
                False,
            )

        # Default low-risk normalization
        return (
            0.15,
            f"Normalized '{from_val}' -> '{to_val}'.",
            "low",
            True,
            False,
        )

    def assess_tabular(
        self,
        raw_df: pl.DataFrame,
        clean_df: pl.DataFrame,
    ) -> LossImpactReport:
        """Run L4 4D loss assessment on tabularized document data."""
        return self.tabular_estimator.assess(raw_df, clean_df)

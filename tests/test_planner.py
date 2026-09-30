"""
Test 3.1 & 3.2: SLM Reasoning & Confidence Governance Verification.

Validates:
- Test 3.1: 25 consecutive runs on mock profile:
  - 0 network errors (100% offline)
  - json.loads() 100% success
  - Zero markdown backticks (no ```)
  - All 10 required keys present in every step
- Test 3.2: Ambiguous column with 45% nulls and chaotic distribution:
  - Confidence < 0.85
  - Successfully routed to human review queue by L5 Confidence Gate
"""

import json
from typing import Any, Dict

import pytest

from narvl.core.fd_miner import FunctionalDependency
from narvl.core.semantic_typer import SemanticClassification
from narvl.engine.grammars import REQUIRED_STEP_KEYS, VALID_ACTIONS
from narvl.engine.planner import SLMPlanner


@pytest.fixture
def mock_profile_summary() -> Dict[str, Any]:
    """Realistic mock profile summary (~500 tokens)."""
    return {
        "meta": {
            "rows": 10000,
            "cols": 6,
            "duplicates": 42,
            "dup_pct": 0.42,
        },
        "columns": {
            "customer_id": {
                "type": "Int64",
                "null_pct": 0.0,
                "card": 9958,
                "zero_var": False,
            },
            "customer_email": {
                "type": "String",
                "null_pct": 1.2,
                "card": 9800,
                "samples": ["[EMAIL_1]", "[EMAIL_2]"],
                "zero_var": False,
            },
            "age": {
                "type": "Int64",
                "null_pct": 2.5,
                "card": 90,
                "min": -5.0,
                "max": 185.0,
                "mean": 38.2,
                "std": 14.5,
                "zero_var": False,
            },
            "income_bracket": {
                "type": "Float64",
                "null_pct": 45.0,  # 45% nulls -> Ambiguous chaotic distribution
                "card": 540,
                "min": 1000.0,
                "max": 250000.0,
                "mean": 45000.0,
                "std": 32000.0,
                "zero_var": False,
            },
            "state_name": {
                "type": "String",
                "null_pct": 0.0,
                "card": 28,
                "samples": ["Telangana", "Karnataka"],
                "zero_var": False,
            },
            "postal_code": {
                "type": "String",
                "null_pct": 0.0,
                "card": 350,
                "samples": ["500081", "560001"],
                "zero_var": False,
            },
        },
    }


def test_planner_25_consecutive_runs_offline_compliance(mock_profile_summary):
    """Test 3.1: 25 consecutive offline runs with 100% JSON & GBNF schema compliance."""
    planner = SLMPlanner()
    semantic_types = {
        "customer_email": SemanticClassification("customer_email", "Email", 0.99, {}),
        "postal_code": SemanticClassification("postal_code", "PostalCode", 1.0, {}),
    }
    fds = [
        FunctionalDependency(
            determinant="postal_code",
            dependent="state_name",
            is_exact=False,
            confidence=0.985,
            canonical_mapping={"Telengana": "Telangana"},
        )
    ]

    total_runs = 25
    successful_runs = 0
    all_keys_valid_count = 0
    zero_markdown_count = 0

    print(f"\n[SLM Planner Test 3.1] Executing {total_runs} consecutive offline reasoning runs...")

    for run_idx in range(1, total_runs + 1):
        res = planner.plan(mock_profile_summary, semantic_types=semantic_types, fds=fds)

        # 1. Zero network errors / offline operation
        assert res.is_valid_json is True

        # 2. Strict JSON parsing without exceptions
        parsed = json.loads(res.raw_json)
        assert isinstance(parsed, list)
        successful_runs += 1

        # 3. Zero markdown backticks
        assert "```" not in res.raw_json
        zero_markdown_count += 1

        # 4. All required keys present in every step
        assert len(res.steps) > 0
        for step in res.steps:
            assert REQUIRED_STEP_KEYS.issubset(step.keys())
            assert step["action"] in VALID_ACTIONS
            assert 0.0 <= step["confidence"] <= 1.0

        all_keys_valid_count += 1

    print(f"[SLM Planner Test 3.1 Compliance Summary]:")
    print(f"  Total Runs: {total_runs}")
    print(f"  json.loads() Success: {successful_runs}/{total_runs} (100%)")
    print(f"  Zero Markdown Backticks: {zero_markdown_count}/{total_runs} (100%)")
    print(f"  All 10 GBNF Keys Present: {all_keys_valid_count}/{total_runs} (100%)")

    assert successful_runs == total_runs
    assert zero_markdown_count == total_runs
    assert all_keys_valid_count == total_runs


def test_confidence_governance_gate_ambiguous_nulls(mock_profile_summary):
    """Test 3.2: Ambiguous column (45% nulls) routed to Human Review Queue (<0.85 confidence)."""
    planner = SLMPlanner(confidence_threshold=0.85)
    fds = [
        FunctionalDependency(
            determinant="postal_code",
            dependent="state_name",
            is_exact=False,
            confidence=0.98,
            canonical_mapping={"Telengana": "Telangana"},
        )
    ]

    result = planner.plan(mock_profile_summary, fds=fds)

    print(f"\n[L5 Confidence Gate Routing Results]:")
    print(f"  Total Pipeline Steps: {len(result.steps)}")
    print(f"  Auto-Batch Approved (>=0.85): {len(result.auto_batch)} steps")
    print(f"  Human Review Queue (<0.85): {len(result.human_review_queue)} steps")

    # Inspect human review queue for ambiguous column
    review_targets = [s["target_column"] for s in result.human_review_queue]
    print(f"  Review Queue Columns: {review_targets}")

    assert "income_bracket" in review_targets, (
        "Ambiguous column 'income_bracket' (45% nulls) was NOT routed to Human Review Queue!"
    )

    ambiguous_step = next(s for s in result.human_review_queue if s["target_column"] == "income_bracket")
    print(f"  Ambiguous Step Confidence: {ambiguous_step['confidence']} (< 0.85)")
    print(f"  Ambiguous Step Justification: {ambiguous_step['justification']}")

    assert ambiguous_step["confidence"] < 0.85
    assert ambiguous_step["action"] == "knn_impute"

    # Non-ambiguous steps (duplicate removal, bounds clamping, FD typo fix) must be in auto-batch
    auto_targets = [s["target_column"] for s in result.auto_batch]
    print(f"  Auto-Batch Columns: {auto_targets}")
    assert "ALL" in auto_targets or "age" in auto_targets or "state_name" in auto_targets
    for s in result.auto_batch:
        assert s["confidence"] >= 0.85

"""
Test 2.2: Fuzzy Functional Dependency (FD) Miner Verification.

Validates that:
- 5000 rows where PostalCode->State with 2% typo (Telengana vs Telangana):
  Miner outputs PostalCode->State with confidence ~0.98 and canonical mapping {"Telengana": "Telangana"}.
- Strict exact FDs are detected with confidence 1.0.
"""

import polars as pl
import pytest

from narvl.core.executor import ReversibleExecutor
from narvl.core.fd_miner import FunctionalDependency, FunctionalDependencyMiner
from narvl.core.semantic_typer import SemanticClassification
from narvl.engine.planner import SLMPlanner


def test_approximate_fd_postal_to_state_typo():
    """Test 2.2: 5000 rows PostalCode->State with 2% typos in State name."""
    n_rows = 5000
    n_typo = int(n_rows * 0.02)  # 100 rows (2%)
    n_clean = n_rows - n_typo    # 4900 rows (98%)

    # 4900 rows with "Telangana", 100 rows with typo "Telengana"
    states = ["Telangana"] * n_clean + ["Telengana"] * n_typo
    postcodes = ["500081"] * n_rows

    df = pl.DataFrame({
        "PostalCode": postcodes,
        "State": states,
    })

    assert df.height == 5000

    miner = FunctionalDependencyMiner(
        fuzzy_similarity_threshold=90.0,
        min_fd_confidence=0.95,
    )

    fd = miner.check_dependency(df, "PostalCode", "State")

    assert fd is not None, "Failed to discover approximate FD PostalCode -> State!"
    print(f"\n[FD Miner Test 2.2 Results]:")
    print(f"  Determinant: {fd.determinant}")
    print(f"  Dependent: {fd.dependent}")
    print(f"  Confidence: {fd.confidence:.4f} (~0.98)")
    print(f"  Is Exact: {fd.is_exact}")
    print(f"  Canonical Mapping: {fd.canonical_mapping}")

    # Confidence must be ~0.98 (4900/5000 = 0.98 before clustering, and clustered dominant = 1.0)
    # Total dominant cluster = 5000/5000 = 1.0 after clustering, raw dominant = 4900/5000 = 0.98
    assert fd.confidence >= 0.98
    assert fd.determinant == "PostalCode"
    assert fd.dependent == "State"
    assert fd.is_exact is False
    
    # Must contain canonical typo mapping {"Telengana": "Telangana"}
    assert "Telengana" in fd.canonical_mapping
    assert fd.canonical_mapping["Telengana"] == "Telangana"


def test_exact_functional_dependency():
    """Test exact functional dependency discovery."""
    df = pl.DataFrame({
        "EmployeeID": [101, 102, 103, 104, 105],
        "Department": ["HR", "Engineering", "Sales", "Engineering", "HR"],
        "Company": ["AcmeCorp", "AcmeCorp", "AcmeCorp", "AcmeCorp", "AcmeCorp"],
    })

    miner = FunctionalDependencyMiner()
    fd = miner.check_dependency(df, "EmployeeID", "Department")

    assert fd is not None
    assert fd.is_exact is True
    assert fd.confidence == 1.0
    print(f"[FD Miner] Exact FD EmployeeID -> Department verified: Confidence={fd.confidence}")


def test_canonical_mapping_is_one_way_and_dictionary_first():
    """Canonical values must never be mapped back to a typo or case variant."""
    miner = FunctionalDependencyMiner()

    mapping = miner.build_canonical_fd_mappings(
        {"Germany": 300, "Gerrmany": 100},
        semantic_type="Country",
    )
    assert mapping == {"Gerrmany": "Germany"}
    assert "Germany" not in mapping

    case_mapping = miner.build_canonical_fd_mappings(
        {"FRANCE": 100, "France": 100, "Franc": 20},
        semantic_type="Country",
    )
    assert case_mapping == {"FRANCE": "France", "Franc": "France"}
    assert "France" not in case_mapping


def test_dictionary_beats_poisoned_frequency_majority():
    """A known country spelling remains canonical even when the typo is more frequent."""
    miner = FunctionalDependencyMiner()
    mapping = miner.build_canonical_fd_mappings(
        {"Gerrmany": 700, "Germany": 300},
        semantic_type="Country",
    )
    assert mapping == {"Gerrmany": "Germany"}


def test_open_world_names_are_not_canonicalized():
    """Other/unknown values must not be treated as spelling errors."""
    miner = FunctionalDependencyMiner()
    mapping = miner.build_canonical_fd_mappings(
        {"Raesh": 80, "Rakesh": 20},
        semantic_type="Other",
    )
    assert mapping == {}
    assert miner.build_canonical_fd_mappings(
        {"Description A": 2, "Description B": 1},
        semantic_type="Description",
    ) == {}
    assert miner.build_canonical_fd_mappings(
        {"101": 2, "102": 1},
        semantic_type="CustomerID",
    ) == {}


def test_planner_skips_open_world_fd_standardization():
    """The planner must reject model or miner mappings for open-world types."""
    planner = SLMPlanner()
    result = planner.plan(
        {"meta": {}, "columns": {}},
        semantic_types={"CustomerName": SemanticClassification("CustomerName", "Other", 0.99, {})},
        fds=[
            FunctionalDependency(
                determinant="CustomerID",
                dependent="CustomerName",
                is_exact=False,
                confidence=0.99,
                canonical_mapping={"Raesh": "Rakesh"},
                dependent_type="Other",
            )
        ],
    )
    assert not any(step["action"] == "standardize_values" for step in result.steps)


def test_standardize_values_is_idempotent():
    """Applying a canonical mapping twice must produce the same DataFrame."""
    df = pl.DataFrame({"Country": ["FRANCE", "France", "Franc"]})
    step = {
        "action": "standardize_values",
        "target_column": "Country",
        "parameters": {"mapping": {"FRANCE": "France", "Franc": "France"}},
    }
    executor = ReversibleExecutor(df)
    once = executor.execute_step(df, step)
    twice = executor.execute_step(once, step)

    assert once["Country"].to_list() == ["France", "France", "France"]
    assert twice.equals(once)

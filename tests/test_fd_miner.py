"""
Test 2.2: Fuzzy Functional Dependency (FD) Miner Verification.

Validates that:
- 5000 rows where PostalCode->State with 2% typo (Telengana vs Telangana):
  Miner outputs PostalCode->State with confidence ~0.98 and canonical mapping {"Telengana": "Telangana"}.
- Strict exact FDs are detected with confidence 1.0.
"""

import polars as pl
import pytest

from narvl.core.fd_miner import FunctionalDependencyMiner


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

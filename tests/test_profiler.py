"""
Test 2.1: Profiler & Semantic Typer Verification.

Validates that:
- 100k-row customer dataset with phones/emails profiles in < 2.5s CPU
- Profile JSON token count is strictly < 2000 tokens
- Semantic Typer classifies Email, City, and PostalCode via ONNX runtime
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from narvl.core.profiler import DatasetProfiler
from narvl.core.semantic_typer import SemanticTyper


def test_100k_profiler_latency_and_token_ceiling():
    """Test 2.1: 100k customer dataset profiling with token ceiling & latency check."""
    n_rows = 100_000
    print(f"\n[Profiler Test 2.1] Generating synthetic 100k dataset...")

    # Generate realistic customer data
    cust_ids = np.arange(1, n_rows + 1)
    ages = np.random.randint(18, 80, size=n_rows).astype(float)
    ages[::20] = np.nan  # 5% nulls

    credit_scores = np.random.normal(700, 50, size=n_rows)
    
    # 100k emails containing @gmail.com, @corporate.com
    emails = [f"user_{i}@gmail.com" if i % 2 == 0 else f"emp.{i}@enterprise.org" for i in range(n_rows)]
    
    # 100k phone numbers (US & India format)
    phones = [f"+91 98765{i % 90000 + 10000:05d}" if i % 2 == 0 else f"555-432-{i % 9000 + 1000:04d}" for i in range(n_rows)]

    cities = ["Hyderabad", "Bengaluru", "Mumbai", "Delhi", "Chennai"] * (n_rows // 5)
    postal_codes = ["500081", "560001", "400001", "110001", "600001"] * (n_rows // 5)
    signup_dates = ["2023-01-15T10:00:00Z", "2023-02-20T12:30:00Z", "2023-03-05T08:15:00Z"] * (n_rows // 3 + 1)
    signup_dates = signup_dates[:n_rows]

    df = pl.DataFrame({
        "customer_id": cust_ids,
        "email": emails,
        "phone_number": phones,
        "age": ages,
        "credit_score": credit_scores,
        "city": cities,
        "postal_code": postal_codes,
        "signup_timestamp": signup_dates,
    })

    assert df.height == 100_000
    assert df.width == 8

    profiler = DatasetProfiler()
    
    # Profile execution
    start_time = time.perf_counter()
    profile = profiler.profile(df)
    elapsed = time.perf_counter() - start_time

    print(f"\n[Profiler Test 2.1 Results]:")
    print(f"  Rows Profiled: {df.height:,}")
    print(f"  Latency on CPU: {elapsed:.3f} seconds (requirement < 2.5s)")
    print(f"  Profile Token Count: {profile.token_count} tokens (requirement < 2000 tokens)")

    # 1. Latency check (< 2.5s)
    assert elapsed < 2.5, f"Profiler latency {elapsed:.3f}s exceeded 2.5s ceiling!"

    # 2. Token count check (< 2000 tokens)
    assert profile.token_count < 2000, f"Summary token count {profile.token_count} exceeded 2000 tokens!"

    # 3. Check samples are captured directly
    email_meta = profile.summary_dict["columns"]["email"]
    assert len(email_meta["samples"]) > 0

    # 4. Check that pattern ratios were properly recorded
    assert "patterns" in email_meta
    assert email_meta["patterns"].get("email_ratio", 0) >= 0.9

    phone_meta = profile.summary_dict["columns"]["phone_number"]
    assert "patterns" in phone_meta
    assert phone_meta["patterns"].get("phone_ratio", 0) >= 0.9


def test_semantic_typer_onnx_classification():
    """Test L2.5 ONNX Semantic Typer classification accuracy on enterprise columns."""
    sample_df = pl.DataFrame({
        "contact_email": ["alice@work.com", "bob@org.net", "charlie@gmail.com"],
        "residence_city": ["Hyderabad", "Bengaluru", "Mumbai"],
        "zip_code": ["500081", "560001", "400001"],
        "account_state": ["Telangana", "Karnataka", "Maharashtra"],
        "transaction_amount": ["$1,500.50", "€200.00", "₹45,000"],
        "event_time": ["2023-05-12T14:30:00Z", "2023-06-14T09:15:00Z", "2023-07-01T18:00:00Z"],
    })

    typer = SemanticTyper(confidence_threshold=0.80)
    results = typer.classify_dataset(sample_df)

    print("\n[Semantic Typer ONNX Classification Results]:")
    for col, res in results.items():
        print(f"  Col '{col}' -> Type: {res.semantic_type} (Confidence: {res.confidence})")

    assert results["contact_email"].semantic_type == "Email"
    assert results["residence_city"].semantic_type == "City"
    assert results["zip_code"].semantic_type == "PostalCode"
    assert results["account_state"].semantic_type == "State"
    assert results["transaction_amount"].semantic_type == "Currency"
    assert results["event_time"].semantic_type == "Timestamp"

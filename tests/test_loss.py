"""
Test 4.1 & 4.2: 4D Loss Estimator & Speculative Utility Barrier Verification.

Validates that:
- Test 4.1: Column X ~ N(50, 10):
  - Strategy A: Median impute 2% nulls -> W1 < 0.05, is_safe=True.
  - Strategy B: Drop values outside [48, 52] -> Loss_vol >= 0.20, W1 > 0.35, is_safe=False -> Safety Gate blocks.
- Test 4.2: Noisy label modification (replacing 60% labels with constant dummy):
  - LightGBM proxy predictive accuracy degrades (utility delta < 0).
  - Safety Gate blocks execution and emits detailed impact report.
"""

import numpy as np
import polars as pl
import pytest

from narvl.core.loss import LossEstimator, LossImpactReport


def test_4d_loss_statistical_w1_and_volumetric_barrier():
    """Test 4.1: Statistical W1 and Volumetric Loss Safety Gate."""
    np.random.seed(42)
    n = 10_000
    # X ~ N(50, 10)
    x_raw = np.random.normal(50.0, 10.0, size=n)
    
    raw_df = pl.DataFrame({
        "id": np.arange(n),
        "X": x_raw,
    })

    estimator = LossEstimator()

    # --- Strategy A: 2% nulls imputed by median ---
    x_nulls = x_raw.copy()
    null_indices = np.random.choice(n, size=int(0.02 * n), replace=False)
    median_val = float(np.median(x_raw))
    x_imputed = x_nulls.copy()
    x_imputed[null_indices] = median_val

    clean_df_a = pl.DataFrame({
        "id": np.arange(n),
        "X": x_imputed,
    })

    report_a = estimator.evaluate_loss_and_safety(raw_df, clean_df_a)
    w1_a = report_a.w1_per_column["X"]

    print(f"\n[Loss Estimator Test 4.1 - Strategy A (Median Impute 2% Nulls)]:")
    print(f"  Volumetric Loss: {report_a.volumetric_loss:.4f} (<= 0.15)")
    print(f"  Statistical W1: {w1_a:.4f} (< 0.05)")
    print(f"  Utility Delta: {report_a.utility_delta:.4f}")
    print(f"  Safety Verdict: is_safe={report_a.is_safe}")

    assert report_a.volumetric_loss == 0.0
    assert w1_a < 0.05, f"W1 distance {w1_a} exceeded 0.05 ceiling!"
    assert report_a.is_safe is True
    assert len(report_a.blocking_reasons) == 0

    # --- Strategy B: Drop values outside [48, 52] ---
    # In N(50, 10), dropping values outside [48, 52] severely truncates the distribution
    clean_df_b = raw_df.filter((pl.col("X") >= 48.0) & (pl.col("X") <= 52.0))
    
    report_b = estimator.evaluate_loss_and_safety(raw_df, clean_df_b)
    w1_b = report_b.w1_per_column["X"]

    print(f"\n[Loss Estimator Test 4.1 - Strategy B (Aggressive Truncation [48, 52])]:")
    print(f"  Volumetric Loss: {report_b.volumetric_loss:.4f} (> 0.15)")
    print(f"  Statistical W1: {w1_b:.4f} (> 0.35)")
    print(f"  Safety Verdict: is_safe={report_b.is_safe}")
    print(f"  Blocking Reasons: {report_b.blocking_reasons}")

    assert report_b.volumetric_loss >= 0.20, f"Volumetric loss {report_b.volumetric_loss} should be >= 0.20"
    assert w1_b > 0.35, f"Statistical W1 {w1_b} should be > 0.35"
    assert report_b.is_safe is False
    assert len(report_b.blocking_reasons) >= 1
    assert any("Volumetric loss" in r or "Wasserstein" in r for r in report_b.blocking_reasons)


def test_speculative_utility_barrier_with_lightgbm_proxy():
    """Test 4.2: Speculative utility degradation blocks bad cleaning plan."""
    np.random.seed(42)
    n = 2000

    # Predictive dataset: X1, X2 determine binary class label Y with high accuracy
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    # True relationship: Y = 1 if x1 + x2 > 0 else 0 (plus 5% noise)
    signal = x1 + x2
    y_raw = (signal > 0).astype(int)
    flip_idx = np.random.choice(n, size=int(0.05 * n), replace=False)
    y_raw[flip_idx] = 1 - y_raw[flip_idx]

    raw_df = pl.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "target": y_raw,
    })

    # Candidate cleaning strategy replaces 60% of labels with a constant 0
    y_corrupted = y_raw.copy()
    corrupt_idx = np.random.choice(n, size=int(0.60 * n), replace=False)
    y_corrupted[corrupt_idx] = 0

    candidate_df = pl.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "target": y_corrupted,
    })

    estimator = LossEstimator()
    report = estimator.evaluate_loss_and_safety(raw_df, candidate_df, target_col="target")

    print(f"\n[Speculative Utility Barrier Test 4.2 Results]:")
    print(f"  Raw LightGBM Utility: {report.raw_utility:.4f}")
    print(f"  Clean LightGBM Utility: {report.clean_utility:.4f}")
    print(f"  Utility Delta: {report.utility_delta:.4f} (< 0.0)")
    print(f"  Safety Gate: is_safe={report.is_safe}")
    print(f"  Blocking Reasons: {report.blocking_reasons}")

    # Assertions
    assert report.utility_delta < 0.0, "Expected negative utility delta from corrupted label cleaning!"
    assert report.is_safe is False, "Safety Gate FAILED to block utility-degrading transformation!"
    assert any("utility degraded" in r.lower() for r in report.blocking_reasons)

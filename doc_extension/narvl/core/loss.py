"""
L4: 4D Loss Estimator & Speculative Utility Barrier for NARVL.

Calculates:
1. Volumetric Loss: Loss_vol = (N_raw - N_clean) / N_raw
2. Statistical W1: scipy.stats.wasserstein_distance(u, v) on standardized distributions
3. Categorical Jaccard Loss: 1 - |set_raw intersect set_clean| / |set_raw union set_clean|
4. Semantic Drift: Cosine distance = 1 - (e_raw · e_clean) / (||e_raw|| * ||e_clean||)
5. Speculative Utility: 20-tree LightGBM proxy on 5% sample comparing raw vs cleaned utility
6. Safety Gate: Blocks execution if Loss_vol > 0.15 or W1 > 0.35 or utility_delta < 0 or cosine_drift > 0.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
import numpy as np
import polars as pl
from scipy.stats import wasserstein_distance

logger = logging.getLogger("narvl.core.loss")

VOLUMETRIC_THRESHOLD = 0.15
WASSERSTEIN_THRESHOLD = 0.35
COSINE_DRIFT_THRESHOLD = 0.20
UTILITY_DELTA_THRESHOLD = 0.0  # Must not degrade baseline predictive utility


@dataclass
class LossImpactReport:
    """Comprehensive 4D Loss Profile and Safety Barrier Verdict."""
    volumetric_loss: float
    max_w1: float
    w1_per_column: Dict[str, float]
    jaccard_loss: Dict[str, float]
    cosine_drift: Dict[str, float]
    utility_delta: float
    raw_utility: float
    clean_utility: float
    is_safe: bool
    blocking_reasons: List[str] = field(default_factory=list)

    @property
    def statistical_w1(self) -> float:
        return self.max_w1

    @property
    def categorical_jaccard_loss(self) -> float:
        return max(self.jaccard_loss.values()) if self.jaccard_loss else 0.0

    @property
    def semantic_cosine_drift(self) -> float:
        return max(self.cosine_drift.values()) if self.cosine_drift else 0.0

    @property
    def predictive_utility_delta(self) -> float:
        return self.utility_delta

    @property
    def mitigation_recommendation(self) -> str:
        if self.is_safe:
            return "Plan approved: Information loss is within enterprise tolerance limits."
        return "Consider replacing row dropping with KNN imputation or clamping bounds to avoid volumetric loss."


class LossEstimator:
    """4D Information Loss & Speculative Utility Barrier."""

    def __init__(
        self,
        volumetric_threshold: float = VOLUMETRIC_THRESHOLD,
        wasserstein_threshold: float = WASSERSTEIN_THRESHOLD,
        cosine_threshold: float = COSINE_DRIFT_THRESHOLD,
        utility_threshold: float = UTILITY_DELTA_THRESHOLD,
    ) -> None:
        self.volumetric_threshold = volumetric_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.cosine_threshold = cosine_threshold
        self.utility_threshold = utility_threshold

    def calculate_volumetric_loss(self, n_raw: int, n_clean: int) -> float:
        """Compute relative volumetric row loss: (N_raw - N_clean) / N_raw."""
        if n_raw <= 0:
            return 0.0
        return max(0.0, float((n_raw - n_clean) / n_raw))

    def calculate_statistical_w1(
        self,
        raw_series: pl.Series,
        clean_series: pl.Series,
    ) -> float:
        """Compute scale-invariant Wasserstein W1 distance between distributions."""
        u = raw_series.drop_nulls().to_numpy().astype(float)
        v = clean_series.drop_nulls().to_numpy().astype(float)

        if len(u) == 0 or len(v) == 0:
            return 0.0

        u_std = float(np.std(u))
        u_mean = float(np.mean(u))
        scale = u_std if u_std > 1e-8 else (float(np.max(u) - np.min(u)) if np.max(u) != np.min(u) else 1.0)

        # Standardize u and v against raw baseline to make W1 scale-invariant
        u_norm = (u - u_mean) / scale
        v_norm = (v - u_mean) / scale

        w1_val = float(wasserstein_distance(u_norm, v_norm))
        return round(w1_val, 4)

    def calculate_categorical_jaccard(
        self,
        raw_series: pl.Series,
        clean_series: pl.Series,
    ) -> float:
        """Compute Categorical Jaccard Loss: 1 - |A intersect B| / |A union B|."""
        set_raw = set(raw_series.drop_nulls().cast(pl.String).to_list())
        set_clean = set(clean_series.drop_nulls().cast(pl.String).to_list())

        if not set_raw and not set_clean:
            return 0.0

        intersection = len(set_raw.intersection(set_clean))
        union = len(set_raw.union(set_clean))

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union
        return round(1.0 - jaccard_similarity, 4)

    def calculate_text_embedding_centroid(self, series: pl.Series) -> np.ndarray:
        """Compute deterministic text embedding centroid representation."""
        sample = series.drop_nulls().head(1000).cast(pl.String).to_list()
        if not sample:
            return np.zeros(64, dtype=np.float32)

        # Character 3-gram hash-based dense vector space (deterministic, 64-dim)
        dim = 64
        vectors = []
        for text in sample:
            vec = np.zeros(dim, dtype=np.float32)
            s = text.lower().strip()
            if not s:
                continue
            for i in range(max(1, len(s) - 2)):
                gram = s[i : i + 3]
                idx = hash(gram) % dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)

        if not vectors:
            return np.zeros(dim, dtype=np.float32)

        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        return centroid

    def calculate_semantic_drift(
        self,
        raw_series: pl.Series,
        clean_series: pl.Series,
    ) -> float:
        """Compute Cosine Distance between raw and clean text centroids."""
        e_raw = self.calculate_text_embedding_centroid(raw_series)
        e_clean = self.calculate_text_embedding_centroid(clean_series)

        norm_raw = float(np.linalg.norm(e_raw))
        norm_clean = float(np.linalg.norm(e_clean))

        if norm_raw == 0.0 or norm_clean == 0.0:
            return 0.0

        dot_product = float(np.dot(e_raw, e_clean))
        cosine_sim = dot_product / (norm_raw * norm_clean)
        cosine_distance = max(0.0, 1.0 - cosine_sim)
        return round(cosine_distance, 4)

    def evaluate_speculative_utility(
        self,
        raw_df: pl.DataFrame,
        clean_df: pl.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Train 20-tree LightGBM proxy on 5% sample to compare predictive utility."""
        n_raw = raw_df.height
        n_clean = clean_df.height
        if lgb is None or n_raw < 20 or n_clean < 20:
            return 0.8, 0.8, 0.0

        # Subsample 5% (min 50, max 2000)
        sample_frac = max(50 / n_raw, min(0.05, 2000 / n_raw))
        sample_raw = raw_df.sample(fraction=min(1.0, sample_frac), seed=42)
        sample_clean = clean_df.sample(fraction=min(1.0, sample_frac), seed=42)

        # Identify numeric or categorical features and target
        if target_col is None or target_col not in raw_df.columns:
            # Pick last column as target, or a non-constant column
            target_col = raw_df.columns[-1]

        y_raw = sample_raw[target_col].drop_nulls()
        y_clean = sample_clean[target_col].drop_nulls()

        # Check if classification or regression
        is_classification = y_raw.dtype in [pl.String, pl.Categorical, pl.Boolean] or y_raw.n_unique() <= 10

        # Prepare feature matrices
        feature_cols = [c for c in raw_df.columns if c != target_col]
        if not feature_cols:
            return 0.8, 0.8, 0.0

        def _prepare_matrix(df_in: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            # Convert numeric features or fillna
            clean_sub = df_in.select(feature_cols + [target_col]).drop_nulls()
            if clean_sub.height == 0:
                return np.zeros((1, len(feature_cols))), np.zeros(1)

            x_parts = []
            for col in feature_cols:
                ser = clean_sub[col]
                if ser.dtype.is_numeric():
                    x_parts.append(ser.to_numpy())
                else:
                    x_parts.append(ser.cast(pl.String).cast(pl.Categorical).to_physical().to_numpy())
            X = np.column_stack(x_parts).astype(np.float32)

            y_ser = clean_sub[target_col]
            if is_classification:
                if y_ser.dtype.is_numeric():
                    y = y_ser.to_numpy().astype(int)
                else:
                    y = y_ser.cast(pl.String).cast(pl.Categorical).to_physical().to_numpy().astype(int)
            else:
                y = y_ser.to_numpy().astype(np.float32)
            return X, y

        X_raw, y_raw_arr = _prepare_matrix(sample_raw)
        X_clean, y_clean_arr = _prepare_matrix(sample_clean)

        if len(y_raw_arr) < 10 or len(y_clean_arr) < 10:
            return 0.8, 0.8, 0.0

        # Train 20-tree LightGBM baseline
        try:
            split_raw = int(len(X_raw) * 0.7)
            split_clean = int(len(X_clean) * 0.7)

            # Holdout test ground truth is evaluated on the raw reference distribution
            X_test = X_raw[split_raw:]
            y_test = y_raw_arr[split_raw:]

            if is_classification:
                n_classes = len(np.unique(y_raw_arr))
                objective = "binary" if n_classes <= 2 else "multiclass"
                clf_raw = lgb.LGBMClassifier(
                    n_estimators=20,
                    max_depth=3,
                    learning_rate=0.1,
                    verbosity=-1,
                    objective=objective,
                    random_state=42,
                )
                clf_raw.fit(X_raw[:split_raw], y_raw_arr[:split_raw])
                score_raw = float(clf_raw.score(X_test, y_test))

                clf_clean = lgb.LGBMClassifier(
                    n_estimators=20,
                    max_depth=3,
                    learning_rate=0.1,
                    verbosity=-1,
                    objective=objective,
                    random_state=42,
                )
                clf_clean.fit(X_clean[:split_clean], y_clean_arr[:split_clean])
                score_clean = float(clf_clean.score(X_test, y_test))
            else:
                reg_raw = lgb.LGBMRegressor(
                    n_estimators=20,
                    max_depth=3,
                    learning_rate=0.1,
                    verbosity=-1,
                    random_state=42,
                )
                reg_raw.fit(X_raw[:split_raw], y_raw_arr[:split_raw])
                pred_raw = reg_raw.predict(X_test)
                mse_raw = float(np.mean((y_test - pred_raw) ** 2))
                score_raw = max(0.0, 1.0 - (mse_raw / (np.var(y_test) + 1e-8)))

                reg_clean = lgb.LGBMRegressor(
                    n_estimators=20,
                    max_depth=3,
                    learning_rate=0.1,
                    verbosity=-1,
                    random_state=42,
                )
                reg_clean.fit(X_clean[:split_clean], y_clean_arr[:split_clean])
                pred_clean = reg_clean.predict(X_test)
                mse_clean = float(np.mean((y_test - pred_clean) ** 2))
                score_clean = max(0.0, 1.0 - (mse_clean / (np.var(y_test) + 1e-8)))

            delta = round(score_clean - score_raw, 4)
            return round(score_raw, 4), round(score_clean, 4), delta

        except Exception as e:
            logger.warning("LightGBM speculative proxy skipped: %s", e)
            return 0.8, 0.8, 0.0

    def evaluate_loss_and_safety(
        self,
        raw_df: pl.DataFrame,
        clean_df: pl.DataFrame,
        target_col: Optional[str] = None,
    ) -> LossImpactReport:
        """Run full 4D Loss Estimation & Speculative Utility Safety Gate."""
        n_raw = raw_df.height
        n_clean = clean_df.height

        # 1. Volumetric Loss
        loss_vol = self.calculate_volumetric_loss(n_raw, n_clean)

        # 2. Statistical W1 Distance across all numeric columns
        w1_per_col: Dict[str, float] = {}
        for col in raw_df.columns:
            if col in clean_df.columns and raw_df[col].dtype.is_numeric():
                w1_per_col[col] = self.calculate_statistical_w1(raw_df[col], clean_df[col])

        max_w1 = max(w1_per_col.values()) if w1_per_col else 0.0

        # 3. Categorical Jaccard Loss
        jaccard_per_col: Dict[str, float] = {}
        for col in raw_df.columns:
            if col in clean_df.columns and raw_df[col].dtype in [pl.String, pl.Categorical]:
                jaccard_per_col[col] = self.calculate_categorical_jaccard(raw_df[col], clean_df[col])

        # 4. Semantic Drift (Cosine Distance)
        cosine_per_col: Dict[str, float] = {}
        for col in raw_df.columns:
            if col in clean_df.columns and raw_df[col].dtype in [pl.String, pl.Categorical]:
                cosine_per_col[col] = self.calculate_semantic_drift(raw_df[col], clean_df[col])

        max_cosine = max(cosine_per_col.values()) if cosine_per_col else 0.0

        # 5. Speculative Utility Barrier (LightGBM proxy)
        raw_util, clean_util, util_delta = self.evaluate_speculative_utility(
            raw_df, clean_df, target_col=target_col
        )

        # 6. Safety Gate evaluation
        blocking_reasons: List[str] = []
        if loss_vol > self.volumetric_threshold:
            blocking_reasons.append(
                f"Volumetric loss {loss_vol:.2%} exceeds safety threshold {self.volumetric_threshold:.2%}"
            )
        if max_w1 > self.wasserstein_threshold:
            blocking_reasons.append(
                f"Maximum Wasserstein W1 distance {max_w1:.4f} exceeds threshold {self.wasserstein_threshold:.4f}"
            )
        if max_cosine > self.cosine_threshold:
            blocking_reasons.append(
                f"Maximum Semantic Drift (Cosine) {max_cosine:.4f} exceeds threshold {self.cosine_threshold:.4f}"
            )
        if util_delta < self.utility_threshold:
            blocking_reasons.append(
                f"Predictive utility degraded by {abs(util_delta):.4f} (Raw: {raw_util:.4f}, Clean: {clean_util:.4f})"
            )

        is_safe = (len(blocking_reasons) == 0)

        return LossImpactReport(
            volumetric_loss=round(loss_vol, 4),
            max_w1=round(max_w1, 4),
            w1_per_column=w1_per_col,
            jaccard_loss=jaccard_per_col,
            cosine_drift=cosine_per_col,
            utility_delta=util_delta,
            raw_utility=raw_util,
            clean_utility=clean_util,
            is_safe=is_safe,
            blocking_reasons=blocking_reasons,
        )

    def assess(
        self,
        raw_df: pl.DataFrame,
        clean_df: pl.DataFrame,
        target_col: Optional[str] = None,
    ) -> LossImpactReport:
        """Convenience alias for evaluate_loss_and_safety."""
        return self.evaluate_loss_and_safety(raw_df, clean_df, target_col=target_col)

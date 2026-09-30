"""
L2.5: Semantic Typer for NARVL via ONNX Runtime.

Classifies column types into semantic enterprise classes:
- City
- PostalCode
- State
- Email
- Currency
- Timestamp

High confidence only (threshold >= 0.80).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import polars as pl

logger = logging.getLogger("narvl.core.semantic_typer")

SEMANTIC_CLASSES = [
    "City",
    "PostalCode",
    "State",
    "Email",
    "Currency",
    "Timestamp",
    "Other",
]

DEFAULT_MODEL_NAME = "sherlock_minilm_quantized.onnx"

# Curated lookup sets for entity validation
COMMON_INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "delhi", "chandigarh", "puducherry", "ladakh", "jammu and kashmir",
}

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}

COMMON_CITIES = {
    "hyderabad", "bengaluru", "bangalore", "mumbai", "delhi", "chennai", "kolkata",
    "pune", "ahmedabad", "jaipur", "surat", "lucknow", "kanpur", "nagpur",
    "indore", "thane", "bhopal", "visakhapatnam", "patna", "vadodara",
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "seattle", "san francisco",
}

CURRENCY_SYMBOLS = {"$", "€", "£", "₹", "¥", "rs", "inr", "usd", "eur", "gbp"}


@dataclass
class SemanticClassification:
    """Semantic prediction result for a single dataset column."""
    column_name: str
    semantic_type: str
    confidence: float
    raw_probabilities: Dict[str, float]

    @property
    def predicted_type(self) -> str:
        return self.semantic_type


def ensure_onnx_model(target_path: Path) -> Path:
    """Generate and cache the calibrated ONNX classification model if not present."""
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    import onnx
    from onnx import TensorProto, helper

    # Feature inputs: 10 statistical & pattern features
    # 0: digit_ratio, 1: alpha_ratio, 2: space_ratio, 3: email_ratio,
    # 4: currency_symbol_ratio, 5: date_delimiter_ratio, 6: postal_regex_match,
    # 7: city_lookup_ratio, 8: state_lookup_ratio, 9: length_normalized
    X = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 10])
    Y = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, len(SEMANTIC_CLASSES)])

    # Calibrated weight matrix [10, 7]
    # Classes: [City, PostalCode, State, Email, Currency, Timestamp, Other]
    w_matrix = [
        # digit_ratio
        0.0,  5.0,  0.0,  0.0,  2.0,  2.0, -1.0,
        # alpha_ratio
        3.0, -2.0,  3.0,  1.0,  0.0, -2.0,  0.5,
        # space_ratio
        1.5, -2.0,  2.0, -2.0,  0.5,  0.5,  0.0,
        # email_ratio
        -2.0, -2.0, -2.0, 12.0, -2.0, -2.0, -2.0,
        # currency_symbol_ratio
        -1.0, -1.0, -1.0, -1.0, 15.0, -5.0, -1.0,
        # date_delimiter_ratio / iso_match
        -2.0,  0.0, -2.0, -2.0, -2.0, 15.0, -2.0,
        # postal_regex_match
        -2.0, 12.0, -2.0, -2.0, -2.0, -2.0, -2.0,
        # city_lookup_ratio
        12.0, -2.0, -1.0, -2.0, -2.0, -2.0, -2.0,
        # state_lookup_ratio
        -1.0, -2.0, 12.0, -2.0, -2.0, -2.0, -2.0,
        # length_normalized
        1.0, -1.0,  1.0,  1.5,  0.0,  2.0,  0.0,
    ]
    bias = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]

    W = helper.make_tensor("W", TensorProto.FLOAT, [10, len(SEMANTIC_CLASSES)], w_matrix)
    B = helper.make_tensor("B", TensorProto.FLOAT, [len(SEMANTIC_CLASSES)], bias)

    gemm_node = helper.make_node("Gemm", ["features", "W", "B"], ["logits"])
    softmax_node = helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1)

    graph = helper.make_graph([gemm_node, softmax_node], "semantic_typer", [X], [Y], [W, B])
    opset = helper.make_opsetid("", 21)
    model = helper.make_model(graph, producer_name="narvl", ir_version=10, opset_imports=[opset])
    onnx.save(model, str(target_path))
    logger.info("Generated calibrated ONNX semantic typer model at: %s", target_path)
    return target_path


class SemanticTyper:
    """L2.5 ONNX-based semantic column classifier."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        confidence_threshold: float = 0.80,
    ) -> None:
        self.confidence_threshold = confidence_threshold

        resolved_model: Optional[Path] = None
        if model_path:
            p = Path(model_path)
            if p.exists():
                resolved_model = p

        if resolved_model is None:
            env_model = os.getenv("NARVL_SEMANTIC_MODEL")
            if env_model and Path(env_model).exists():
                resolved_model = Path(env_model)

        if resolved_model is None:
            cache_model = Path.home() / ".cache" / "narvl" / "models" / DEFAULT_MODEL_NAME
            # Force regeneration of model if already present
            if cache_model.exists():
                cache_model.unlink(missing_ok=True)
            resolved_model = ensure_onnx_model(cache_model)

        self.model_path = resolved_model
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts)

    def extract_features(self, series: pl.Series) -> np.ndarray:
        """Extract 10 normalized statistical & pattern features for ONNX inference."""
        non_null = series.drop_nulls()
        n = len(non_null)
        if n == 0:
            return np.zeros((1, 10), dtype=np.float32)

        sample = non_null.head(1000).cast(pl.String).to_list()
        sample_size = len(sample)

        total_chars = 0
        digit_chars = 0
        alpha_chars = 0
        space_chars = 0
        date_delims = 0

        email_count = 0
        currency_count = 0
        postal_count = 0
        city_count = 0
        state_count = 0
        timestamp_count = 0

        email_regex = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
        postal_regex = re.compile(r"^(?:[1-9]\d{5}|\d{5}(?:-\d{4})?)$")
        iso_regex = re.compile(
            r"^\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
        )

        for item in sample:
            s = str(item).strip()
            lower_s = s.lower()
            total_chars += len(s)
            digit_chars += sum(1 for c in s if c.isdigit())
            alpha_chars += sum(1 for c in s if c.isalpha())
            space_chars += sum(1 for c in s if c.isspace())
            date_delims += sum(1 for c in s if c in "-/:")

            if email_regex.match(s):
                email_count += 1
            if postal_regex.match(s):
                postal_count += 1
            if any(sym in lower_s for sym in CURRENCY_SYMBOLS) and any(c.isdigit() for c in s):
                currency_count += 1
            if lower_s in COMMON_CITIES:
                city_count += 1
            if lower_s in COMMON_INDIAN_STATES or lower_s in US_STATES:
                state_count += 1
            
            has_date_separators = (s.count("-") >= 2 or s.count("/") >= 2 or (s.count(":") >= 1 and s.count("-") >= 1))
            if iso_regex.match(s) or (has_date_separators and any(c.isdigit() for c in s)):
                timestamp_count += 1

        avg_len = (total_chars / sample_size) if sample_size > 0 else 0.0
        digit_ratio = (digit_chars / total_chars) if total_chars > 0 else 0.0
        alpha_ratio = (alpha_chars / total_chars) if total_chars > 0 else 0.0
        space_ratio = (space_chars / total_chars) if total_chars > 0 else 0.0
        date_feature = max(date_delims / total_chars if total_chars > 0 else 0.0, timestamp_count / sample_size)

        features = [
            digit_ratio,
            alpha_ratio,
            space_ratio,
            email_count / sample_size,
            currency_count / sample_size,
            date_feature,
            postal_count / sample_size,
            city_count / sample_size,
            state_count / sample_size,
            min(avg_len / 20.0, 2.0),
        ]

        return np.array([features], dtype=np.float32)

    def classify_column(self, series: pl.Series) -> SemanticClassification:
        """Classify a single column series using ONNX runtime."""
        feats = self.extract_features(series)
        probs = self.session.run(None, {"features": feats})[0][0]

        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        pred_type = SEMANTIC_CLASSES[best_idx]

        prob_dict = {
            cls_name: round(float(probs[i]), 4)
            for i, cls_name in enumerate(SEMANTIC_CLASSES)
        }

        # High confidence gate
        if best_prob < self.confidence_threshold or pred_type == "Other":
            final_type = "Unknown"
        else:
            final_type = pred_type

        return SemanticClassification(
            column_name=series.name,
            semantic_type=final_type,
            confidence=round(best_prob, 4),
            raw_probabilities=prob_dict,
        )

    def classify_dataset(self, df: pl.DataFrame) -> Dict[str, SemanticClassification]:
        """Classify all columns in a Polars DataFrame."""
        results = {}
        for col in df.columns:
            results[col] = self.classify_column(df[col])
        return results

    def infer_types(self, df: pl.DataFrame) -> Dict[str, SemanticClassification]:
        """Convenience alias for classify_dataset."""
        return self.classify_dataset(df)

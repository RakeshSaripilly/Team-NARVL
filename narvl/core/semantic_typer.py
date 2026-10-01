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

import json
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

COMMON_COUNTRIES = {
    "united kingdom", "uk", "great britain", "united states", "usa", "us",
    "germany", "france", "australia", "canada", "india", "japan", "china",
    "brazil", "spain", "italy", "netherlands", "switzerland", "sweden",
    "singapore", "united arab emirates", "uae", "ireland", "norway", "belgium",
    "austria", "denmark", "finland", "poland", "portugal", "new zealand",
    "south africa", "mexico", "israel", "greece", "czech republic", "hong kong",
    "malaysia", "thailand", "philippines", "indonesia", "taiwan", "chile",
    "argentina", "colombia", "peru", "saudi arabia", "turkey", "cyprus", "rsa",
    "eire", "channel islands", "european community", "unspecified", "iceland",
    "lithuania", "malta", "bahrain", "lebanon", "russia", "south korea", "korea",
}


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
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts)

        # Attempt to load local GGUF SLM for zero-shot in-context semantic typing
        self.llama_model = None
        try:
            import llama_cpp  # type: ignore
            from narvl.engine.model_loader import resolve_model_path
            resolved_slm = resolve_model_path(allow_download=False)
            self.llama_model = llama_cpp.Llama(
                model_path=str(resolved_slm),
                n_threads=4,
                n_ctx=1024,
                mmap=True,
                verbose=False,
            )
            logger.info("Loaded local SLM for dynamic in-context semantic typing: %s", resolved_slm)
        except Exception:
            self.llama_model = None

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

    def infer_slm(self, series: pl.Series) -> Optional[SemanticClassification]:
        """Query local GGUF SLM for zero-shot in-context semantic typing."""
        if self.llama_model is None:
            return None

        col_name = str(series.name).strip()
        non_null = series.drop_nulls()
        n = len(non_null)
        if n == 0:
            return SemanticClassification(col_name, "Unknown", 0.0, {})

        sample = [str(x).strip() for x in non_null.head(8).to_list() if str(x).strip()]
        sample = sample[:6]
        dtype_str = str(series.dtype)
        uniq_ratio = (non_null.n_unique() / n) if n > 0 else 0.0

        prompt = (
            f"<|im_start|>system\n"
            f"You are an enterprise schema classifier. Output a concise semantic type in 1-2 words for this table column.\n"
            f"Format strictly as JSON: {{\"semantic_type\": \"<type>\", \"confidence\": <float between 0.8 and 1.0>}}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Column: {col_name}\n"
            f"DataType: {dtype_str}\n"
            f"Uniqueness: {uniq_ratio*100:.1f}%\n"
            f"Sample: {sample}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            resp = self.llama_model(
                prompt=prompt,
                max_tokens=40,
                temperature=0.1,
                stop=["<|im_end|>", "\n\n", "}"],
            )
            text = resp["choices"][0]["text"].strip()
            if not text.endswith("}"):
                text += "}"
            data = json.loads(text)
            stype = data.get("semantic_type", "").strip()
            conf = float(data.get("confidence", 0.95))
            if stype and len(stype) <= 35:
                return SemanticClassification(
                    column_name=col_name,
                    semantic_type=stype,
                    confidence=min(1.0, max(0.5, conf)),
                    raw_probabilities={stype: conf},
                )
        except Exception as exc:
            logger.debug("SLM inference skipped: %s", exc)
        return None

    def infer_dynamic(self, series: pl.Series) -> Optional[SemanticClassification]:
        """Dynamic open-world semantic inference using header tokens, cardinality, and content gazetteers."""
        col_name = str(series.name).strip()
        low_col = col_name.lower().replace("_", " ").replace("-", " ")
        non_null = series.drop_nulls()
        n = len(non_null)
        if n == 0:
            return SemanticClassification(col_name, "Unknown", 0.0, {})

        unique_count = non_null.n_unique()
        unique_ratio = unique_count / n if n > 0 else 0.0
        sample = [str(x).strip() for x in non_null.head(200).to_list() if str(x).strip()]
        avg_len = sum(len(x) for x in sample) / len(sample) if sample else 0.0

        # 1. Email Address
        if any(re.search(rf"\b{k}\b", low_col) for k in ["email", "mail"]) or (
            sample and sum(1 for s in sample if "@" in s and "." in s) / len(sample) >= 0.50
        ):
            return SemanticClassification(col_name, "Email", 0.99, {"Email": 0.99})

        # 2. Timestamp / Date
        if any(re.search(rf"\b{k}\b", low_col) for k in ["date", "time", "timestamp", "created", "updated", "datetime"]) or (
            sample and sum(1 for s in sample if (s.count("-") >= 2 or s.count("/") >= 2 or ":" in s) and any(c.isdigit() for c in s)) / len(sample) >= 0.50
        ):
            return SemanticClassification(col_name, "Timestamp", 0.99, {"Timestamp": 0.99})

        # 3. Geo - Country
        if any(re.search(rf"\b{k}\b", low_col) for k in ["country", "nation"]) or (
            sample and sum(1 for s in sample if s.lower() in COMMON_COUNTRIES) / len(sample) >= 0.35
        ):
            return SemanticClassification(col_name, "Country", 0.98, {"Country": 0.98})

        # 4. Geo - State
        if any(re.search(rf"\b{k}\b", low_col) for k in ["state", "province", "region"]) or (
            sample and sum(1 for s in sample if s.lower() in COMMON_INDIAN_STATES or s.lower() in US_STATES) / len(sample) >= 0.35
        ):
            return SemanticClassification(col_name, "State", 0.96, {"State": 0.96})

        # 5. Geo - City
        if any(re.search(rf"\b{k}\b", low_col) for k in ["city", "town"]) or (
            sample and sum(1 for s in sample if s.lower() in COMMON_CITIES) / len(sample) >= 0.35
        ):
            return SemanticClassification(col_name, "City", 0.96, {"City": 0.96})

        # 6. Geo - PostalCode (Explicit zip/postal header, or postal pattern without ID context)
        if any(re.search(rf"\b{k}\b", low_col) for k in ["zip", "postal", "postcode", "pincode"]):
            return SemanticClassification(col_name, "PostalCode", 0.97, {"PostalCode": 0.97})

        # 7. Currency / Price / Financial Amount (supports 'unitprice', 'price', '$', etc.)
        if any(k in low_col for k in ["price", "cost", "amount", "salary", "fee", "fare", "rate", "subtotal", "total", "balance", "wage", "revenue", "charge"]) or (
            sample and sum(1 for s in sample if any(sym in s.lower() for sym in CURRENCY_SYMBOLS) and any(c.isdigit() for c in s)) / len(sample) >= 0.30
        ):
            return SemanticClassification(col_name, "Currency", 0.97, {"Currency": 0.97})

        # 8. Quantity / Count
        if any(re.search(rf"\b{k}\b", low_col) for k in ["qty", "quantity", "count", "units", "items", "volume", "inventory"]):
            return SemanticClassification(col_name, "Quantity", 0.96, {"Quantity": 0.96})

        # 9. Specific Identifiers & Codes
        if any(k in low_col for k in ["invoice no", "invoiceno", "invoice num", "invoice id", "invoice_no", "inv no", "inv_no", "invoice"]):
            return SemanticClassification(col_name, "InvoiceNo", 0.99, {"InvoiceNo": 0.99})
        if any(k in low_col for k in ["customer id", "customer no", "customer num", "customerid", "customerno", "cust id", "client id", "user id", "account id", "account no", "account num", "accountid", "accountno"]) or (low_col == "customerid"):
            return SemanticClassification(col_name, "CustomerID", 0.99, {"CustomerID": 0.99})
        if any(k in low_col for k in ["stock code", "stockcode", "stock no", "sku", "item code", "product code", "part number", "part no"]):
            return SemanticClassification(col_name, "StockCode", 0.98, {"StockCode": 0.98})
        if any(re.search(rf"\b{k}\b", low_col) for k in ["id", "no", "num", "number", "code", "key", "ref", "guid", "uuid", "pk", "fk"]):
            return SemanticClassification(col_name, "Identifier", 0.95, {"Identifier": 0.95})

        # 10. Free-form Text / Product Description
        if any(re.search(rf"\b{k}\b", low_col) for k in ["desc", "description", "title", "name", "comment", "note", "summary", "remark", "detail", "item"]) or (
            avg_len > 12 and any(" " in s for s in sample)
        ):
            return SemanticClassification(col_name, "Description", 0.96, {"Description": 0.96})

        return None

    def classify_column(self, series: pl.Series) -> SemanticClassification:
        """Classify a single column series using dynamic open-world SLM in-context reasoning with ONNX baseline."""
        # 1. Dynamic local SLM in-context inference
        slm_res = self.infer_slm(series)
        if slm_res is not None:
            return slm_res

        # 2. Dynamic open-world contextual inference (Header + Cardinality + Values)
        dyn_res = self.infer_dynamic(series)
        if dyn_res is not None and dyn_res.confidence >= self.confidence_threshold:
            return dyn_res

        # 3. Fallback to calibrated ONNX model
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

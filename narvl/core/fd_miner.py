"""
L3: Fuzzy Functional Dependency (FD) Miner for NARVL.

Implements:
1. Exact FD:
   df.group_by(col_a).agg(pl.col(col_b).n_unique()) -> if max==1 strict FD
2. Approximate & Fuzzy FD:
   group_by X, collect target Y values, cluster via rapidfuzz.fuzz.ratio > 90,
   compute Confidence = sum(max count per X) / N.
   If Confidence >= 0.98, emit approximate FD with canonical typo repair mapping.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import pycountry
from rapidfuzz import fuzz

logger = logging.getLogger("narvl.core.fd_miner")

def _normalise_label(value: str) -> str:
    return " ".join(str(value).casefold().replace("-", " ").split())


def _build_pycountry_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for country in pycountry.countries:
        canonical = getattr(country, "common_name", country.name)
        for attr in ("name", "official_name", "common_name", "alpha_2", "alpha_3"):
            value = getattr(country, attr, None)
            if value:
                lookup[_normalise_label(value)] = canonical
    for subdivision in pycountry.subdivisions:
        lookup.setdefault(_normalise_label(subdivision.name), subdivision.name)
    return lookup


PYCOUNTRY_LOOKUP = _build_pycountry_lookup()


@dataclass
class FunctionalDependency:
    """Discovered Functional Dependency between determinant X and dependent Y."""
    determinant: str  # X
    dependent: str    # Y
    is_exact: bool
    confidence: float
    canonical_mapping: Dict[str, str] = field(default_factory=dict)
    sample_violations: List[Dict[str, Any]] = field(default_factory=list)
    dependent_type: str = "Unknown"
    canonical_confidence: float = 1.0
    dependent_cardinality_ratio: float = 1.0

    def is_closed_world(self) -> bool:
        return self.dependent_cardinality_ratio < 0.05


class FunctionalDependencyMiner:
    """Discovers exact and approximate functional dependencies across dataset columns."""

    def __init__(
        self,
        fuzzy_similarity_threshold: float = 88.0,
        min_fd_confidence: float = 0.98,
        max_cardinality_ratio: float = 0.95,
        fuzzy_threshold: Optional[float] = None,
        semantic_types: Optional[Dict[str, Any]] = None,
        enable_slm_resolution: bool = False,
    ) -> None:
        if fuzzy_threshold is not None:
            self.fuzzy_similarity_threshold = fuzzy_threshold
            if min_fd_confidence == 0.98:
                self.min_fd_confidence = 0.85
            else:
                self.min_fd_confidence = min_fd_confidence
        else:
            self.fuzzy_similarity_threshold = fuzzy_similarity_threshold
            self.min_fd_confidence = min_fd_confidence
        self.max_cardinality_ratio = max_cardinality_ratio
        self.semantic_types = semantic_types or {}
        self.enable_slm_resolution = enable_slm_resolution
        self.global_canonical_set: set[str] = set()

    def _resolve_semantic_type(
        self,
        column_name: str,
        series: Optional[pl.Series] = None,
        semantic_types: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolve a dependent column type without guessing open-world names."""
        supplied = (semantic_types or self.semantic_types).get(column_name)
        if supplied is not None:
            resolved = getattr(supplied, "semantic_type", supplied)
            if resolved:
                return str(resolved)

        if series is not None:
            values = {str(value).strip().casefold() for value in series.drop_nulls().head(1000)}
            if values and all(value in PYCOUNTRY_LOOKUP for value in values):
                return "ClosedWorld"

        return "Unknown"

    def _lookup_for_type(self, semantic_type: str) -> Dict[str, str]:
        return PYCOUNTRY_LOOKUP

    def choose_canonical_with_dict(
        self,
        cluster_values: List[str],
        value_counts: Dict[str, int],
        semantic_type: str,
    ) -> Tuple[str, bool]:
        """Choose a canonical value dynamically.

        Priority:
        1. pycountry lookup as optional tie-breaker (not sole source)
        2. Dynamic selection: max by (frequency, istitle, length)
        """
        # Dynamic canonical: highest frequency, then title-case preference, then longest
        dynamic_canonical = max(
            cluster_values,
            key=lambda v: (
                value_counts.get(v, 0),
                str(v).istitle(),
                len(str(v)),
            ),
        )

        # Use pycountry as optional tie-breaker only when dynamic choice is ambiguous
        lookup = self._lookup_for_type(semantic_type)
        for value in cluster_values:
            normalized = _normalise_label(value)
            if normalized in lookup:
                dict_canonical = lookup[normalized]
                # Only prefer dict if it matches one of the cluster values (no fabrication)
                for cv in cluster_values:
                    if fuzz.ratio(cv.casefold(), dict_canonical.casefold()) >= 88:
                        return cv, True
                # Dict canonical not in cluster — fall through to dynamic
                break

        return dynamic_canonical, False

    def resolve_canonical_with_slm(
        self,
        cluster_values: List[str],
        semantic_type: str,
    ) -> Optional[Tuple[str, float]]:
        """Optionally resolve a closed-world cluster using only its unique values."""
        if not self.enable_slm_resolution:
            return None

        try:
            from narvl.engine.model_loader import load_llama_model, resolve_model_path

            model_path = resolve_model_path(allow_download=False)
            model = load_llama_model(model_path=model_path, n_ctx=512)
            prompt = (
                f"Pick the correct canonical spelling for {semantic_type}. "
                f"Only choose from or normalize this unique values list: {cluster_values}. "
                'Return JSON only: {"canonical": "...", "confidence": 0.0}. '
                "Do not infer personal names or use row data."
            )
            result = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            payload = json.loads(result["choices"][0]["message"]["content"])
            canonical = str(payload.get("canonical", "")).strip()
            confidence = float(payload.get("confidence", 0.0))
            lookup = self._lookup_for_type(semantic_type)
            is_similar = any(fuzz.ratio(canonical.casefold(), value.casefold()) >= 75 for value in cluster_values)
            if confidence >= 0.85 and (is_similar or canonical.casefold() in lookup):
                return canonical, confidence
        except Exception as exc:
            logger.debug("Closed-world SLM canonical resolver unavailable: %s", exc)
        return None

    def cluster_and_canonicalize(
        self,
        value_counts: Dict[str, int],
        semantic_type: str = "Unknown",
    ) -> Tuple[Dict[str, str], int, float]:
        """Cluster closed-world values and map variants only toward one canonical root.
        
        Args:
            value_counts: Dict mapping raw Y value to occurrence frequency.
            
        Returns:
            Tuple of (canonical_mapping_dict, max_clustered_count, mapping_confidence).
        """
        if not value_counts:
            return {}, 0, 0.6

        clusters: List[List[str]] = []
        for value in value_counts:
            value_text = str(value).strip()
            for cluster in clusters:
                leader = cluster[0]
                similarity = fuzz.ratio(value_text.casefold(), leader.casefold())
                if similarity >= self.fuzzy_similarity_threshold or similarity >= 88.0:
                    cluster.append(value_text)
                    break
            else:
                clusters.append([value_text])

        mapping: Dict[str, str] = {}
        mapping_confidence = 1.0
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            canonical, from_dictionary = self.choose_canonical_with_dict(cluster, value_counts, semantic_type)
            slm_choice = self.resolve_canonical_with_slm(cluster, semantic_type)
            if slm_choice is not None:
                canonical, slm_confidence = slm_choice
                mapping_confidence = min(mapping_confidence, slm_confidence)
            elif not from_dictionary:
                mapping_confidence = min(mapping_confidence, 0.6)

            if from_dictionary:
                mapping_confidence = min(mapping_confidence, 0.92)
            self.global_canonical_set.add(canonical)
            canonical_casefold = canonical.casefold()
            canonical_count = value_counts.get(canonical, 0)
            for member in cluster:
                if member == canonical or member in self.global_canonical_set:
                    continue
                if (
                    member.casefold() == canonical_casefold
                    and member.istitle()
                    and canonical.isupper()
                ):
                    continue
                if not from_dictionary and value_counts.get(member, 0) > canonical_count * 1.5:
                    continue
                mapping[member] = canonical

        return mapping, sum(value_counts.get(value, 0) for value in value_counts), mapping_confidence

    def build_canonical_fd_mappings(
        self,
        value_counts: Dict[str, int],
        semantic_type: str,
        cardinality_ratio: float = 1.0,
    ) -> Dict[str, str]:
        """Build one-way canonical mappings for a closed-world dependent column.

        Closed-world is determined by cardinality ratio < 0.05 (not by semantic type name).
        Open-world columns (ratio > 0.20) are skipped entirely.
        """
        # Skip open-world columns regardless of semantic type label
        if cardinality_ratio > 0.20:
            return {}
        # Skip columns that are clearly not closed-world categorical
        if semantic_type in {"Email", "Timestamp", "Description"}:
            return {}
        total = sum(value_counts.values())
        if not total:
            return {}
        mapping, _, _ = self.cluster_and_canonicalize(value_counts, semantic_type)
        return mapping

    def check_dependency(
        self,
        df: pl.DataFrame,
        col_x: str,
        col_y: str,
        semantic_types: Optional[Dict[str, Any]] = None,
    ) -> Optional[FunctionalDependency]:
        """Test whether determinant col_x functionally determines dependent col_y."""
        clean_df = df.select([col_x, col_y]).drop_nulls()
        n = clean_df.height
        if n == 0:
            return None

        dependent_type = self._resolve_semantic_type(
            col_y,
            clean_df[col_y],
            semantic_types=semantic_types,
        )
        dependent_cardinality_ratio = clean_df[col_y].n_unique() / n
        closed_world = dependent_cardinality_ratio < 0.05

        # 1. Exact FD check via Polars group_by
        grouped = clean_df.group_by(col_x).agg(
            pl.col(col_y).n_unique().alias("n_unq")
        )
        max_unq = grouped["n_unq"].max()
        if max_unq == 1:
            return FunctionalDependency(
                determinant=col_x,
                dependent=col_y,
                is_exact=True,
                confidence=1.0,
                canonical_mapping={},
                sample_violations=[],
                dependent_type=dependent_type,
                dependent_cardinality_ratio=dependent_cardinality_ratio,
            )

        # 2. Approximate / Fuzzy FD check
        # Fuzzy string typo clustering only applies to string/categorical text columns
        y_dtype = clean_df[col_y].dtype
        if y_dtype.is_numeric() or y_dtype == pl.Boolean or y_dtype.is_temporal():
            return None

        # Group by (X, Y) and count occurrences
        xy_pairs = (
            clean_df.group_by([col_x, col_y])
            .len()
            .rename({"len": "count"})
            .sort([col_x, "count", col_y], descending=[False, True, False])
        )

        # Organize by X: {x_val: {y_val: count}}
        x_groups: Dict[Any, Dict[str, int]] = defaultdict(dict)
        for row in xy_pairs.iter_rows():
            x_val, y_val, cnt = row[0], str(row[1]), row[2]
            x_groups[x_val][y_val] = cnt

        total_dominant_count = 0
        overall_canonical_map: Dict[str, str] = {}
        sample_violations: List[Dict[str, Any]] = []
        mapping_confidence = 1.0

        for x_val, y_counts in x_groups.items():
            if len(y_counts) == 1:
                # Single value under x_val
                only_y, cnt = next(iter(y_counts.items()))
                total_dominant_count += cnt
            else:
                # Multiple Y values -> Cluster fuzzy variants (e.g. Telengana vs Telangana)
                if not closed_world:
                    # Open-world column (cardinality > 0.20): skip fuzzy clustering
                    if dependent_cardinality_ratio > 0.20:
                        total_dominant_count += max(y_counts.values())
                        if len(sample_violations) < 5:
                            sample_violations.append({
                                "determinant_val": str(x_val),
                                "observed_variants": list(y_counts.keys()),
                                "counts": y_counts,
                            })
                        continue
                    # Mid-range cardinality: still skip fuzzy clustering but count dominant
                    total_dominant_count += max(y_counts.values())
                    if len(sample_violations) < 5:
                        sample_violations.append({
                            "determinant_val": str(x_val),
                            "observed_variants": list(y_counts.keys()),
                            "counts": y_counts,
                        })
                    continue

                local_map, max_cluster_cnt, local_confidence = self.cluster_and_canonicalize(
                    y_counts,
                    semantic_type=dependent_type,
                )
                total_dominant_count += max_cluster_cnt
                overall_canonical_map.update(local_map)
                mapping_confidence = min(mapping_confidence, local_confidence)

                if len(sample_violations) < 5:
                    sample_violations.append({
                        "determinant_val": str(x_val),
                        "observed_variants": list(y_counts.keys()),
                        "counts": y_counts,
                    })

        confidence = total_dominant_count / n
        if confidence >= self.min_fd_confidence:
            has_repairs = bool(overall_canonical_map)
            return FunctionalDependency(
                determinant=col_x,
                dependent=col_y,
                is_exact=(confidence == 1.0 and not has_repairs),
                confidence=round(confidence, 4),
                canonical_mapping=overall_canonical_map,
                sample_violations=sample_violations,
                dependent_type=dependent_type,
                canonical_confidence=mapping_confidence,
                dependent_cardinality_ratio=dependent_cardinality_ratio,
            )

        return None

    def mine_dependencies(
        self,
        df: pl.DataFrame,
        candidate_columns: Optional[List[str]] = None,
        semantic_types: Optional[Dict[str, Any]] = None,
    ) -> List[FunctionalDependency]:
        """Mine all exact and approximate functional dependencies in the dataset.
        
        Args:
            df: Normalized Polars DataFrame.
            candidate_columns: Optional subset of columns to evaluate.
            
        Returns:
            List of discovered FunctionalDependency objects.
        """
        cols = candidate_columns or df.columns
        n_rows = df.height
        if n_rows == 0:
            return []

        self.global_canonical_set = set()

        # Filter out unique primary keys (cardinality == n_rows) or constant columns
        valid_cols = []
        for col in cols:
            card = df[col].n_unique()
            if 1 < card < (n_rows * self.max_cardinality_ratio):
                valid_cols.append(col)

        dependencies: List[FunctionalDependency] = []

        # Pairwise search
        for col_x in valid_cols:
            for col_y in cols:
                if col_x == col_y:
                    continue
                fd = self.check_dependency(df, col_x, col_y, semantic_types=semantic_types)
                if fd is not None:
                    dependencies.append(fd)

        return dependencies

    def mine(
        self,
        df: pl.DataFrame,
        candidate_columns: Optional[List[str]] = None,
        semantic_types: Optional[Dict[str, Any]] = None,
    ) -> List[FunctionalDependency]:
        """Convenience alias for mine_dependencies."""
        return self.mine_dependencies(
            df,
            candidate_columns=candidate_columns,
            semantic_types=semantic_types,
        )

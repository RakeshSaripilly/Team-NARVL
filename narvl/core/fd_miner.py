"""
L3: Fuzzy Functional Dependency (FD) Miner for NARVL.

Implements:
1. Exact FD:
   df.group_by(col_a).agg(pl.col(col_b).n_unique()) -> if max==1 strict FD
2. Approximate & Fuzzy FD:
   group_by X, collect target Y values, cluster via rapidfuzz.fuzz.ratio > 90,
   resolve canonical entities using global frequency and gazetteers,
   compute Confidence = sum(max count per X) / N.
   If Confidence >= 0.85, emit approximate FD with canonical typo repair mapping.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl
from rapidfuzz import fuzz

from narvl.core.semantic_typer import (
    COMMON_CITIES,
    COMMON_COUNTRIES,
    COMMON_INDIAN_STATES,
    US_STATES,
)

logger = logging.getLogger("narvl.core.fd_miner")

CANONICAL_CASE_MAP: Dict[str, str] = {
    s.lower(): s.title()
    for s in (COMMON_INDIAN_STATES | US_STATES | COMMON_CITIES | COMMON_COUNTRIES)
}
CANONICAL_CASE_MAP["uk"] = "United Kingdom"
CANONICAL_CASE_MAP["u.k."] = "United Kingdom"
CANONICAL_CASE_MAP["great britain"] = "United Kingdom"
CANONICAL_CASE_MAP["usa"] = "USA"
CANONICAL_CASE_MAP["uae"] = "UAE"
CANONICAL_CASE_MAP["rsa"] = "RSA"

KNOWN_CANONICAL_ENTITIES: Set[str] = set(CANONICAL_CASE_MAP.keys())

DATE_PATTERN = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})"
    r"(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
)


def is_temporal_column(ser: pl.Series) -> bool:
    """Check if a series is temporal either by physical dtype or string datetime pattern."""
    if ser.dtype.is_temporal():
        return True
    c_lower = ser.name.lower()
    if any(k in c_lower for k in ["date", "time", "timestamp", "dob", "year", "month"]):
        return True
    non_null = ser.drop_nulls()
    if len(non_null) == 0:
        return False
    sample = [str(x).strip() for x in non_null.head(30) if str(x).strip()]
    if not sample:
        return False
    matches = sum(1 for s in sample if DATE_PATTERN.match(s))
    return (matches / len(sample)) >= 0.5


def diff_involves_digits(s1: str, s2: str) -> bool:
    """Check if differing characters between two strings involve numeric digits."""
    d1 = "".join(c for c in s1 if c.isdigit())
    d2 = "".join(c for c in s2 if c.isdigit())
    return d1 != d2


def are_typo_variants(s1: str, s2: str, fuzzy_threshold: float = 85.0) -> bool:
    """Evaluate whether s1 and s2 are typographical variants of each other."""
    s1_str = str(s1).strip()
    s2_str = str(s2).strip()
    if s1_str == s2_str:
        return True
    # Digits, dates, or IDs differing by numbers are distinct entities, not spelling typos
    if diff_involves_digits(s1_str, s2_str):
        return False
    s1_l = s1_str.lower()
    s2_l = s2_str.lower()
    if s1_l == s2_l:
        return True
    sim = fuzz.ratio(s1_l, s2_l)
    if sim >= fuzzy_threshold or sim >= 85.0:
        return True
    min_len = min(len(s1_l), len(s2_l))
    if min_len >= 4 and (s1_l.startswith(s2_l) or s2_l.startswith(s1_l)):
        return True
    elif abs(len(s1_str) - len(s2_str)) <= 1 and min_len >= 4:
        diffs = sum(c1 != c2 for c1, c2 in zip(s1_l, s2_l))
        if diffs <= 1:
            return True
    return False


@dataclass
class FunctionalDependency:
    """Discovered Functional Dependency between determinant X and dependent Y."""
    determinant: str  # X
    dependent: str    # Y
    is_exact: bool
    confidence: float
    canonical_mapping: Dict[str, str] = field(default_factory=dict)
    sample_violations: List[Dict[str, Any]] = field(default_factory=list)


class FunctionalDependencyMiner:
    """Discovers exact and approximate functional dependencies across dataset columns."""

    def __init__(
        self,
        fuzzy_similarity_threshold: float = 90.0,
        min_fd_confidence: float = 0.98,
        max_cardinality_ratio: float = 0.95,
        fuzzy_threshold: Optional[float] = None,
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

    def cluster_and_canonicalize(
        self,
        value_counts: Dict[str, int],
    ) -> Tuple[Dict[str, str], int]:
        """Cluster strings and select canonical root based on gazetteer and dominance."""
        if not value_counts:
            return {}, 0

        def _candidate_rank(item: Tuple[str, int]) -> Tuple[int, int, int, int, str]:
            val, cnt = item
            clean_str = str(val).strip().lower()
            is_known = 1 if clean_str in KNOWN_CANONICAL_ENTITIES else 0
            is_title = 1 if str(val).istitle() else 0
            return (is_known, cnt, is_title, len(clean_str), str(val))

        sorted_items = sorted(value_counts.items(), key=_candidate_rank, reverse=True)
        clusters: List[List[Tuple[str, int]]] = []
        mapping: Dict[str, str] = {}

        for val, count in sorted_items:
            assigned = False
            for cluster in clusters:
                leader_val, _ = cluster[0]
                if are_typo_variants(val, leader_val, self.fuzzy_similarity_threshold):
                    cluster.append((val, count))
                    mapping[val] = leader_val
                    assigned = True
                    break
            if not assigned:
                clusters.append([(val, count)])
                mapping[val] = val

        cluster_totals = [sum(cnt for _, cnt in cl) for cl in clusters]
        max_clustered_count = max(cluster_totals) if cluster_totals else 0

        # Resolve known canonical casing if available
        resolved_mapping: Dict[str, str] = {}
        for k, v in mapping.items():
            canonical_v = CANONICAL_CASE_MAP.get(v.lower(), v)
            if k != canonical_v:
                resolved_mapping[k] = canonical_v

        return resolved_mapping, max_clustered_count

    def check_dependency(
        self,
        df: pl.DataFrame,
        col_x: str,
        col_y: str,
    ) -> Optional[FunctionalDependency]:
        """Test whether determinant col_x functionally determines dependent col_y."""
        clean_df = df.select([col_x, col_y]).drop_nulls()
        n = clean_df.height
        if n == 0:
            return None

        # Continuous floating point numbers are metrics, not discrete determinants
        if clean_df[col_x].dtype.is_float():
            return None

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
            )

        # 2. Approximate / Fuzzy FD check
        # Fuzzy string typo clustering only applies to text/categorical columns (never numeric or temporal)
        y_ser = clean_df[col_y]
        if y_ser.dtype.is_numeric() or y_ser.dtype == pl.Boolean or is_temporal_column(y_ser):
            return None

        # Global value counts for col_y to ensure globally grounded canonical selection
        global_counts: Dict[str, int] = defaultdict(int)
        for row in y_ser.value_counts().to_dicts():
            global_counts[str(row[col_y])] = row["count"]

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

        # Identify candidate typo variants that co-occur under the same determinant X
        co_variants: Dict[str, Set[str]] = defaultdict(set)
        sample_violations: List[Dict[str, Any]] = []

        for x_val, y_counts in x_groups.items():
            if len(y_counts) > 1:
                vals = list(y_counts.keys())
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        v1, v2 = vals[i], vals[j]
                        if are_typo_variants(v1, v2, self.fuzzy_similarity_threshold):
                            co_variants[v1].add(v2)
                            co_variants[v2].add(v1)

                if len(sample_violations) < 5:
                    sample_violations.append({
                        "determinant_val": str(x_val),
                        "observed_variants": vals,
                        "counts": y_counts,
                    })

        # Find connected components of typo variants and assign global canonical root
        visited = set()
        overall_canonical_map: Dict[str, str] = {}

        for node in list(co_variants.keys()):
            if node not in visited:
                comp: List[str] = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop()
                    comp.append(curr)
                    for neigh in co_variants[curr]:
                        if neigh not in visited:
                            visited.add(neigh)
                            queue.append(neigh)

                # Rank candidates in component globally:
                # 1. Known gazetteer entity
                # 2. Global frequency in the entire dataset
                # 3. Standard Title Case
                # 4. Length
                def _component_rank(val: str) -> Tuple[int, int, int, int]:
                    c_lower = str(val).strip().lower()
                    is_known = 1 if c_lower in KNOWN_CANONICAL_ENTITIES else 0
                    g_cnt = global_counts.get(val, 0)
                    is_title = 1 if str(val).istitle() else 0
                    return (is_known, g_cnt, is_title, len(c_lower))

                best_root = max(comp, key=_component_rank)
                canonical_target = CANONICAL_CASE_MAP.get(best_root.lower(), best_root)
                for v in comp:
                    if v != canonical_target:
                        overall_canonical_map[v] = canonical_target

        # Calculate total dominant count after canonical mapping
        total_dominant_count = 0
        for x_val, y_counts in x_groups.items():
            if len(y_counts) == 1:
                total_dominant_count += next(iter(y_counts.values()))
            else:
                clustered = defaultdict(int)
                for y_val, cnt in y_counts.items():
                    root = overall_canonical_map.get(y_val, y_val)
                    clustered[root] += cnt
                total_dominant_count += max(clustered.values())

        confidence = total_dominant_count / n
        if confidence >= self.min_fd_confidence:
            return FunctionalDependency(
                determinant=col_x,
                dependent=col_y,
                is_exact=(confidence == 1.0),
                confidence=round(confidence, 4),
                canonical_mapping=overall_canonical_map,
                sample_violations=sample_violations,
            )

        return None

    def mine_dependencies(
        self,
        df: pl.DataFrame,
        candidate_columns: Optional[List[str]] = None,
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

        # Filter out unique primary keys (cardinality == n_rows), constant columns, and floats
        valid_cols = []
        for col in cols:
            ser = df[col]
            card = ser.n_unique()
            if ser.dtype.is_float():
                continue
            if 1 < card < (n_rows * self.max_cardinality_ratio):
                valid_cols.append(col)

        dependencies: List[FunctionalDependency] = []

        # Pairwise search
        for col_x in valid_cols:
            for col_y in cols:
                if col_x == col_y:
                    continue
                fd = self.check_dependency(df, col_x, col_y)
                if fd is not None:
                    dependencies.append(fd)

        return dependencies

    def mine(
        self,
        df: pl.DataFrame,
        candidate_columns: Optional[List[str]] = None,
    ) -> List[FunctionalDependency]:
        """Convenience alias for mine_dependencies."""
        return self.mine_dependencies(df, candidate_columns=candidate_columns)

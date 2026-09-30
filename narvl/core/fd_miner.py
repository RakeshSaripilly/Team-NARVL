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

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from rapidfuzz import fuzz

logger = logging.getLogger("narvl.core.fd_miner")


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
        else:
            self.fuzzy_similarity_threshold = fuzzy_similarity_threshold
        self.min_fd_confidence = min_fd_confidence
        self.max_cardinality_ratio = max_cardinality_ratio

    def cluster_and_canonicalize(
        self,
        value_counts: Dict[str, int],
    ) -> Tuple[Dict[str, str], int]:
        """Cluster strings with rapidfuzz.ratio > 90 or edit distance <= 1 and select dominant canonical root.
        
        Args:
            value_counts: Dict mapping raw Y value to occurrence frequency.
            
        Returns:
            Tuple of (canonical_mapping_dict, max_clustered_count).
        """
        if not value_counts:
            return {}, 0

        # Sort values by frequency descending (dominant candidate is first)
        sorted_items = sorted(value_counts.items(), key=lambda kv: kv[1], reverse=True)
        clusters: List[List[Tuple[str, int]]] = []
        mapping: Dict[str, str] = {}

        for val, count in sorted_items:
            assigned = False
            for cluster in clusters:
                # Compare against canonical cluster leader (highest frequency)
                leader_val, _ = cluster[0]
                s1 = str(val).strip()
                s2 = str(leader_val).strip()
                sim = fuzz.ratio(s1.lower(), s2.lower())
                # 1-edit typo (e.g. Telengana vs Telangana = 88.9%) or ratio >= threshold
                is_similar = (sim >= self.fuzzy_similarity_threshold or sim >= 85.0)
                if not is_similar and abs(len(s1) - len(s2)) <= 1:
                    # Check 1-char substitution/insertion
                    diffs = sum(c1 != c2 for c1, c2 in zip(s1.lower(), s2.lower()))
                    if diffs <= 1:
                        is_similar = True

                if is_similar:
                    cluster.append((val, count))
                    mapping[val] = leader_val
                    assigned = True
                    break
            if not assigned:
                clusters.append([(val, count)])
                mapping[val] = val

        # Calculate max clustered count across clusters
        cluster_totals = [sum(cnt for _, cnt in cl) for cl in clusters]
        max_clustered_count = max(cluster_totals) if cluster_totals else 0

        # Only retain non-identity mappings in canonical mapping
        canonical_typo_fixes = {k: v for k, v in mapping.items() if k != v}

        return canonical_typo_fixes, max_clustered_count

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
        # Group by (X, Y) and count occurrences
        xy_pairs = (
            clean_df.group_by([col_x, col_y])
            .len()
            .rename({"len": "count"})
            .sort([col_x, "count"], descending=[False, True])
        )

        # Organize by X: {x_val: {y_val: count}}
        x_groups: Dict[Any, Dict[str, int]] = defaultdict(dict)
        for row in xy_pairs.iter_rows():
            x_val, y_val, cnt = row[0], str(row[1]), row[2]
            x_groups[x_val][y_val] = cnt

        total_dominant_count = 0
        overall_canonical_map: Dict[str, str] = {}
        sample_violations: List[Dict[str, Any]] = []

        for x_val, y_counts in x_groups.items():
            if len(y_counts) == 1:
                # Single value under x_val
                only_y, cnt = next(iter(y_counts.items()))
                total_dominant_count += cnt
            else:
                # Multiple Y values -> Cluster fuzzy variants (e.g. Telengana vs Telangana)
                local_map, max_cluster_cnt = self.cluster_and_canonicalize(y_counts)
                total_dominant_count += max_cluster_cnt
                overall_canonical_map.update(local_map)

                if len(sample_violations) < 5:
                    sample_violations.append({
                        "determinant_val": str(x_val),
                        "observed_variants": list(y_counts.keys()),
                        "counts": y_counts,
                    })

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

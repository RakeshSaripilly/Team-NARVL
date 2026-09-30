"""
Relation Detector and Entity Graph builder for document intelligence.
Detects:
- CustomerName -> Phone, Email, Address, City, State entity clusters
- Proximity-based co-occurrence within sections, paragraphs, or table rows
- Emits structured EntityGraph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from narvl.document.models import ParsedDocument, StructuredEntity


@dataclass
class EntityRelation:
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 0.90


@dataclass
class EntityGraph:
    nodes: Dict[str, StructuredEntity] = field(default_factory=dict)
    edges: List[EntityRelation] = field(default_factory=list)
    clusters: List[Dict[str, Any]] = field(default_factory=list)


class RelationDetector:
    """Discovers relationships between extracted entities and builds an EntityGraph."""

    def __init__(self, proximity_char_window: int = 350) -> None:
        self.proximity_char_window = proximity_char_window

    def detect_relations(
        self,
        entities: List[StructuredEntity],
        doc: Optional[ParsedDocument] = None,
    ) -> EntityGraph:
        """Link related entities (e.g., Customer -> Phone/Email/Address/City) into an EntityGraph."""
        graph = EntityGraph()
        for ent in entities:
            graph.nodes[ent.entity_id] = ent

        # 1. Group entities by record_id when available (blocks / table rows)
        rec_groups: Dict[str, List[StructuredEntity]] = {}
        unassigned: List[StructuredEntity] = []

        for ent in entities:
            if ent.record_id:
                rec_groups.setdefault(ent.record_id, []).append(ent)
            else:
                unassigned.append(ent)

        for rec_id, r_ents in rec_groups.items():
            names = [e for e in r_ents if e.type == "CustomerName"]
            attributes = [e for e in r_ents if e.type in ["Phone", "Email", "Address", "City", "State", "InvoiceNo", "Date", "Amount", "PostalCode"]]

            if names:
                for attr in attributes:
                    best_name = names[0]
                    if len(names) > 1:
                        attr_mid = (attr.source_span[0] + attr.source_span[1]) / 2.0
                        best_name = min(names, key=lambda n: abs(attr_mid - (n.source_span[0] + n.source_span[1]) / 2.0))
                    rel_type = f"has_{attr.type.lower()}"
                    graph.edges.append(
                        EntityRelation(
                            source_id=best_name.entity_id,
                            target_id=attr.entity_id,
                            relation_type=rel_type,
                            confidence=0.98,
                        )
                    )
                    if attr.entity_id not in best_name.relationships:
                        best_name.relationships.append(attr.entity_id)
                    if best_name.entity_id not in attr.relationships:
                        attr.relationships.append(best_name.entity_id)
            else:
                unassigned.extend(attributes)

        # 2. Handle unassigned entities via page proximity
        if unassigned:
            page_groups: Dict[int, List[StructuredEntity]] = {}
            for ent in unassigned:
                page_groups.setdefault(ent.source_page, []).append(ent)

            for page_num, p_ents in page_groups.items():
                names = [e for e in entities if e.source_page == page_num and e.type == "CustomerName"]
                attributes = [e for e in p_ents if e.type in ["Phone", "Email", "Address", "City", "State", "InvoiceNo", "Date", "Amount", "PostalCode"]]

                if len(names) == 1:
                    name_ent = names[0]
                    for attr in attributes:
                        rel_type = f"has_{attr.type.lower()}"
                        graph.edges.append(
                            EntityRelation(
                                source_id=name_ent.entity_id,
                                target_id=attr.entity_id,
                                relation_type=rel_type,
                                confidence=0.95,
                            )
                        )
                        if attr.entity_id not in name_ent.relationships:
                            name_ent.relationships.append(attr.entity_id)
                        if name_ent.entity_id not in attr.relationships:
                            attr.relationships.append(name_ent.entity_id)
                elif len(names) > 1:
                    for attr in attributes:
                        attr_mid = (attr.source_span[0] + attr.source_span[1]) / 2.0
                        best_name: Optional[StructuredEntity] = None
                        min_dist = float("inf")

                        for n in names:
                            n_mid = (n.source_span[0] + n.source_span[1]) / 2.0
                            dist = abs(attr_mid - n_mid)
                            if dist < min_dist and dist <= self.proximity_char_window:
                                min_dist = dist
                                best_name = n

                        if best_name is not None:
                            rel_type = f"has_{attr.type.lower()}"
                            graph.edges.append(
                                EntityRelation(
                                    source_id=best_name.entity_id,
                                    target_id=attr.entity_id,
                                    relation_type=rel_type,
                                    confidence=0.88,
                                )
                            )
                            if attr.entity_id not in best_name.relationships:
                                best_name.relationships.append(attr.entity_id)
                            if best_name.entity_id not in attr.relationships:
                                attr.relationships.append(best_name.entity_id)

        # 3. Build clusters
        cluster_map: Dict[str, Dict[str, Any]] = {}
        for edge in graph.edges:
            parent = graph.nodes.get(edge.source_id)
            child = graph.nodes.get(edge.target_id)
            if parent and child:
                if parent.entity_id not in cluster_map:
                    cluster_map[parent.entity_id] = {
                        "cluster_id": parent.entity_id,
                        "primary_entity": parent.normalized_value or parent.value,
                        "type": parent.type,
                        "record_id": parent.record_id,
                        "attributes": {},
                    }
                cluster_map[parent.entity_id]["attributes"].setdefault(child.type, []).append(
                    child.normalized_value or child.value
                )

        # Ensure any CustomerName without attributes still forms a cluster
        for ent in entities:
            if ent.type == "CustomerName" and ent.entity_id not in cluster_map:
                cluster_map[ent.entity_id] = {
                    "cluster_id": ent.entity_id,
                    "primary_entity": ent.normalized_value or ent.value,
                    "type": ent.type,
                    "record_id": ent.record_id,
                    "attributes": {},
                }

        graph.clusters = list(cluster_map.values())
        return graph

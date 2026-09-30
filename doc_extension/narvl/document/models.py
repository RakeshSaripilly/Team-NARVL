"""
Pydantic schemas and data models for Unstructured Document Intelligence.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    PDF = "PDF"
    DOCX = "DOCX"
    TXT = "TXT"


class Page(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_num: int
    text: str
    tables: List[Any] = Field(default_factory=list)
    char_count: int = 0
    is_scanned: bool = False


class DocumentStructure(BaseModel):
    headings: List[Dict[str, Any]] = Field(default_factory=list)
    sections: List[Dict[str, Any]] = Field(default_factory=list)


class ExtractedTable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    table_id: str
    data: Any  # Polars DataFrame or list of dicts
    source_page: int = 1
    bbox: Optional[Tuple[float, float, float, float]] = None


class ParsedDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str
    original_path: str
    file_type: DocumentType
    text: str
    pages: List[Page] = Field(default_factory=list)
    structure: DocumentStructure = Field(default_factory=DocumentStructure)
    tables: List[ExtractedTable] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


EntityType = Literal[
    "CustomerName",
    "Phone",
    "Email",
    "Date",
    "Address",
    "InvoiceNo",
    "Amount",
    "City",
    "State",
    "PostalCode",
]


class StructuredEntity(BaseModel):
    entity_id: str
    type: EntityType
    value: str
    normalized_value: Optional[str] = None
    source_span: Tuple[int, int] = (0, 0)
    source_page: int = 1
    record_id: Optional[str] = None
    confidence: float = 1.0
    relationships: List[str] = Field(default_factory=list)


QualityIssueType = Literal[
    "missing",
    "duplicate",
    "format_inconsistent",
    "conflict",
    "invalid",
    "semantic",
]


class QualityIssue(BaseModel):
    issue_id: str
    type: QualityIssueType
    field: str
    current_value: Optional[str] = None
    conflicting_values: List[str] = Field(default_factory=list)
    evidence: str = ""
    severity: Literal["low", "medium", "high"] = "low"
    source_page: int = 1


OperationType = Literal["normalize", "merge", "deduplicate", "standardize", "fill"]


class CleaningRecommendation(BaseModel):
    issue_id: str
    explanation: str
    proposed_change: Dict[str, Any] = Field(default_factory=dict)  # {"from": ..., "to": ...}
    operation_type: OperationType
    confidence: float = 1.0
    risk_level: Literal["low", "medium", "high"] = "low"
    loss_score: float = 0.0
    loss_reason: str = ""
    is_reversible: bool = True
    requires_approval: bool = False


class DocumentVersion(BaseModel):
    version_id: int
    parent_id: Optional[int] = None
    operation: str
    timestamp: str
    data_snapshot_path: str
    entities: List[StructuredEntity] = Field(default_factory=list)
    is_reversible: bool = True

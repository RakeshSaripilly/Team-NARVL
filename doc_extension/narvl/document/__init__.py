"""
NARVL Unstructured Document Intelligence Module.
"""

from narvl.document.approval_manager import ApprovalManager
from narvl.document.cleaning_planner import DocumentCleaningPlanner
from narvl.document.docx_extractor import DocxExtractor
from narvl.document.entity_extractor import EntityExtractor
from narvl.document.exporters import DocumentExporter
from narvl.document.file_handler import DocumentFileHandler
from narvl.document.loss_estimator import DocumentLossEstimator
from narvl.document.models import (
    CleaningRecommendation,
    DocumentStructure,
    DocumentType,
    DocumentVersion,
    ExtractedTable,
    Page,
    ParsedDocument,
    QualityIssue,
    StructuredEntity,
)
from narvl.document.ocr_engine import OCREngine
from narvl.document.pdf_extractor import PdfExtractor
from narvl.document.pipeline import (
    DocumentCleaningPipeline,
    entities_to_dataframe,
    entities_to_records_dataframe,
)
from narvl.document.quality_analyzer import QualityAnalyzer
from narvl.document.relation_detector import EntityGraph, RelationDetector
from narvl.document.txt_extractor import TxtExtractor
from narvl.document.validator import DocumentValidator, ValidationReport

__all__ = [
    "ApprovalManager",
    "CleaningRecommendation",
    "DocxExtractor",
    "DocumentCleaningPipeline",
    "DocumentCleaningPlanner",
    "DocumentExporter",
    "DocumentFileHandler",
    "DocumentLossEstimator",
    "DocumentStructure",
    "DocumentType",
    "DocumentValidator",
    "DocumentVersion",
    "EntityExtractor",
    "EntityGraph",
    "ExtractedTable",
    "OCREngine",
    "Page",
    "ParsedDocument",
    "PdfExtractor",
    "QualityAnalyzer",
    "QualityIssue",
    "RelationDetector",
    "StructuredEntity",
    "TxtExtractor",
    "ValidationReport",
    "entities_to_dataframe",
    "entities_to_records_dataframe",
]

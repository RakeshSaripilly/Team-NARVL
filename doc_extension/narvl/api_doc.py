"""
FastAPI REST API for Document Intelligence Module.
Provides:
- POST /api/v1/clean-document (multipart file upload)
- POST /api/v1/rollback-document (reversible time-travel rollback)
- GET /healthz
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from narvl.document.pipeline import DocumentCleaningPipeline

app = FastAPI(
    title="NARVL Document Intelligence API",
    description="Offline Autonomous Document Ingestion, Quality Analysis, and Reversible Cleaning",
    version="0.1.0",
)

pipeline = DocumentCleaningPipeline()


class RollbackDocumentRequest(BaseModel):
    doc_id: str
    target_version: int = 0


@app.get("/healthz")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "module": "narvl-document-intelligence",
        "supported_formats": ["TXT", "DOCX", "PDF"],
    }


@app.post("/api/v1/clean-document")
async def clean_document_endpoint(
    file: UploadFile = File(...),
    auto_approve: bool = Query(False, description="Auto-approve recommendations"),
) -> Dict[str, Any]:
    """Upload and process an unstructured document through the NARVL cleaning pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".txt", ".docx", ".pdf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Supported formats: .txt, .docx, .pdf",
        )

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
        content = await file.read()
        tmp_f.write(content)
        tmp_path = Path(tmp_f.name)

    try:
        res = pipeline.run_cleaning_flow(tmp_path, auto_approve=auto_approve)
        records_df = res["records_dataframe"]

        return {
            "status": "success",
            "doc_id": res["doc_id"],
            "filename": file.filename,
            "document_type": res["file_type"],
            "raw_entities_count": res["raw_entity_count"],
            "cleaned_entities_count": res["cleaned_entity_count"],
            "quality_issues": res["quality_issues"],
            "recommendations": res["recommendations"],
            "governance_summary": res["governance_summary"],
            "validation_report": res["validation_report"],
            "delta_versions": [v["version_id"] for v in res["versions"]],
            "cleaned_records": records_df.to_dicts(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/v1/rollback-document")
def rollback_document_endpoint(payload: RollbackDocumentRequest) -> Dict[str, Any]:
    """Roll back a document dataset to a previous Delta Lake version."""
    try:
        restored = pipeline.rollback_to_version(payload.doc_id, payload.target_version)
        return {
            "status": "success",
            "doc_id": payload.doc_id,
            "restored_version": payload.target_version,
            "entity_count": len(restored),
            "restored_entities": [e.model_dump() for e in restored],
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

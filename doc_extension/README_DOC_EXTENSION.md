# NARVL Unstructured Document Intelligence Extension (`doc_extension/`)

Autonomous, offline-first, and reversible data cleaning extension for unstructured documents (**PDF, DOCX, and TXT**). Built as an isolated module within the NARVL L0–L6 autonomy stack without modifying any root files.

---

## 1. Architectural Overview

Messy unstructured documents (invoices, reports, customer account logs) are parsed, analyzed for data quality anomalies, planned with information loss bounds, and cleaned via an immutable, reversible Delta Lake DAG:

```
PDF / DOCX / TXT Document
          ↓
[Stage 1] L0 Streaming Shield & File Preservation (data/originals/)
          ↓
[Stage 2] Multi-Format Document Parser (Text + Headings + Tables)
          ↓
[Stage 3] Semantic Entity Extractor & Relation Graph (L2.5 ONNX + Regex)
          ↓
[Stage 4] Data Quality Detection (6 Issue Classes: Missing, Duplicate, Inconsistent, Conflict, Invalid, Semantic)
          ↓
[Stage 5] Cleaning Planner & L4 4D Information Loss Barrier (Volumetric, W1, Jaccard, Cosine, Predictive Utility)
          ↓
[Stage 6] L5 Confidence Governance Gate (Auto-Batch vs Human Review Queue)
          ↓
[Stage 7] L6 Reversible Delta Lake Execution Engine & Dual Validation (Pandera + Great Expectations)
          ↓
Structured Outputs (CSV, JSON, Multi-Sheet Excel, Markdown Report, Signed Provenance Certificate)
```

---

## 2. L0–L6 Autonomy Stack Reuse

All logic reuses and extends the core NARVL architecture:

| Level | Component | Extension Adaptation |
|---|---|---|
| **L0** | **Streaming Adversarial Shield** | 64KB binary chunk streaming, 20MB per-document limit, `\x00` neutralization, and preservation of raw inputs in `doc_extension/data/originals/`. |
| **L1** | **Format Normalizer** | Extraction bridge converting extracted `StructuredEntity` objects and `ExtractedTable` into standard Polars DataFrames. |
| **L2** | **Vectorized Profiler** | Statistical sparsity detection identifying missing required fields (null%) in document clusters. |
| **L2.5**| **ONNX Semantic Typer & FD Miner** | RapidFuzz similarity clustering (ratio ≥ 85% / edit distance ≤ 1) resolving semantic variations (`Hyd`, `HYD`, `Hyderabad` -> `Hyderabad`; `Telengana` -> `Telangana`). |
| **L3** | **Constrained Reasoning Planner** | GBNF-constrained planning schema generating standardized DAG actions (`standardize`, `normalize`, `deduplicate`, `merge`, `fill`). |
| **L4** | **4D Loss Estimator** | Stratified risk scoring: Low-risk normalizations (0.10–0.20 score, reversible) vs High-risk conflicting merges (0.90 score, **BLOCKED**). |
| **L5** | **Confidence Governance Gate & Validator** | Routes confidence ≥ 0.85 & low-risk to Auto-Batch; blocks high-risk in Human Review Queue. Auto-synthesizes 6+ dual validation checks. |
| **L6** | **Reversible Delta Execution** | Step-by-step transaction logs (`_delta_log/`) and time-travel rollback (`rollback_to_version(v)`) with 100% bitwise parity. |

---

## 3. Subfolder Isolation & Zero-Impact Guarantee

This extension strictly obeys **Subfolder Isolation**:
- **Root Repository Untouched**: `narvl/`, `tests/`, `data/`, `pyproject.toml`, `README.md`, `Dockerfile` in root remain completely unmodified.
- **Isolated Workspace**: All new source code, tests, fixtures, and generated artifacts reside strictly inside `doc_extension/`.
- **Self-Contained Execution**: All modules are importable and runnable independently using `doc_extension` on `PYTHONPATH`.

---

## 4. Directory Structure

```
doc_extension/
├── README_DOC_EXTENSION.md           # This comprehensive guide
├── pyproject_doc.toml                # Extension package specification
├── requirements_doc.txt              # Core requirements (Polars, PyMuPDF, python-docx, etc.)
├── requirements_ocr.txt              # Optional OCR requirements (pytesseract, pdf2image)
├── demo_document_flow.py             # End-to-end autonomous demonstration script
├── demo_outputs/                     # Generated deliverables
│   ├── cleaned_data.csv              # Structured cleaned records (CSV)
│   ├── cleaned_data.json             # Structured cleaned records (JSON)
│   ├── cleaned_data.xlsx             # Multi-sheet Excel workbook (Entities, Records, Issues, Audit)
│   ├── cleaning_report.md            # Executive summary markdown report
│   ├── transformation_history.json   # Immutable version commit history
│   ├── validation_results.json       # Detailed validation check verdicts
│   ├── narvl_document_provenance.json# HMAC-SHA256 signed provenance manifest
│   └── narvl_document_provenance.html# Executive HTML provenance audit certificate
├── data/
│   ├── originals/                    # Secure copies of raw ingested documents
│   ├── fixtures/documents/           # Test fixtures (messy_customers.txt, messy_invoice.docx, messy_report.pdf)
│   └── delta_document/               # Delta Lake table store with version snapshots and _delta_log
├── narvl/
│   ├── core/                         # Copied and extended L0-L6 core modules
│   ├── engine/                       # Copied SLM model loader, planner, and grammars
│   ├── document/                     # 15 Document Intelligence modules
│   │   ├── __init__.py
│   │   ├── models.py                 # Pydantic schemas (ParsedDocument, StructuredEntity, QualityIssue, etc.)
│   │   ├── file_handler.py           # Ingestion, preservation, and L0 Shield enforcement
│   │   ├── txt_extractor.py          # Plain text parser with heading & table detection
│   │   ├── docx_extractor.py         # python-docx parser preserving paragraph IDs
│   │   ├── pdf_extractor.py          # PyMuPDF + pdfplumber parser with scanned detection
│   │   ├── ocr_engine.py             # Graceful fallback OCR engine
│   │   ├── entity_extractor.py       # Deterministic & regex entity extraction
│   │   ├── relation_detector.py      # Entity proximity clustering and EntityGraph builder
│   │   ├── quality_analyzer.py       # 6 issue detectors (missing, duplicate, format, conflict, invalid, semantic)
│   │   ├── cleaning_planner.py       # Recommendation generator with risk levels
│   │   ├── loss_estimator.py         # L4 4D Loss & doc-specific risk stratification
│   │   ├── approval_manager.py       # L5 Governance Gate review queue manager
│   │   ├── pipeline.py               # Reversible execution pipeline & Delta versioning
│   │   ├── validator.py              # Automated dual validation engine
│   │   └── exporters.py              # Multi-format artifact exporters & signed provenance
│   ├── cli_doc.py                    # clean-document command-line interface
│   ├── api_doc.py                    # FastAPI REST API (/api/v1/clean-document)
│   └── ui_doc/                       # Streamlit CleanPilot web UI with Screen 0 Document Upload
└── tests/
    └── test_document/                # 9 comprehensive automated pytest suites
```

---

## 5. Quickstart & How to Run

### Set PYTHONPATH
From the repository root on Windows PowerShell:
```powershell
$env:PYTHONPATH = "c:\Team-NARVL\doc_extension"
```

### 1. Run the End-to-End Demo Script
Executes the full pipeline on `messy_customers.txt`, displays the 7 stages, verifies reversibility, and exports all deliverables:
```powershell
python doc_extension/demo_document_flow.py
```

### 2. Run the CLI Tool (`clean-document`)
Clean any messy document directly from the command line:
```powershell
python doc_extension/narvl/cli_doc.py doc_extension/data/fixtures/documents/messy_customers.txt --auto-approve
```

### 3. Launch the REST API
Start the FastAPI server on port 8000:
```powershell
python -m uvicorn narvl.api_doc:app --host 0.0.0.0 --port 8000
```
Upload and clean a document via cURL / PowerShell:
```powershell
curl -X POST "http://localhost:8000/api/v1/clean-document?auto_approve=true" -F "file=@doc_extension/data/fixtures/documents/messy_customers.txt"
```

### 4. Launch CleanPilot Document Web UI
```powershell
streamlit run doc_extension/narvl/ui_doc/app.py
```

### 5. Run Automated Tests
Run all 12 test cases across 9 test modules:
```powershell
python -m pytest doc_extension/tests/ -v
```

---

## 6. Supported Document Types & Features

| Format | Extraction Engine | Extracted Features |
|---|---|---|
| **TXT** | Custom Regex Parser | Paragraphs, double-newline sections, ALL CAPS headings, and delimited tables. |
| **DOCX** | `python-docx` | Headings (H1–H3), paragraph ID provenance tracking, and document tables to Polars. |
| **PDF** | `PyMuPDF` (`fitz`) + `pdfplumber` | Font-size typographic headings, multi-page layout, tables to Polars, and scanned page detection. |
| **Scanned** | `pytesseract` + `pdf2image` | Optional OCR fallback when average characters per page < 50 (offline safe). |

---

## 7. Data Quality Anomaly Detection (6 Detectors)

1. **Missing**: Required fields missing in entity clusters (e.g. customer record without phone or email).
2. **Duplicate**: Exact matches or RapidFuzz near-duplicates (similarity ≥ 85% or edit distance ≤ 1).
3. **Format Inconsistent**: Unstandardized phone formats (raw digits vs hyphens vs international) and date formats (`DD/MM/YYYY`, `MM-DD-YYYY` -> `YYYY-MM-DD`).
4. **Conflicting**: Contradictory values for the same logical customer entity across sources.
5. **Invalid**: RFC email structure failures and phone length violations.
6. **Semantic**: Synonym and regional spelling variations canonicalized to standard roots (`Hyd` / `HYD` -> `Hyderabad`; `Telengana` -> `Telangana`).

---

## 8. Information Loss & Reversibility Guarantee

- **L4 Loss Barrier**: Prevents destructive operations. High-risk conflicting merges receive a 0.90 loss score and are strictly **BLOCKED**.
- **100% Bitwise Parity Rollback**: Every state change commits a Delta snapshot (`_delta_log/`). Rolling back to Version 0 restores the initial extracted state with zero data corruption.

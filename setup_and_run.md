# ⚡ NARVL: Setup & Run Guide

This document provides step-by-step instructions to set up the environment, install dependencies, and run the **NARVL** autonomous data cleaning platform across different modes (Web UI, CLI, and REST API).

---

## 📋 Prerequisites

- **Python**: Version `3.10` to `3.13` (64-bit)
- **Package Manager**: `pip` (bundled with Python)
- **OS**: Windows, macOS, or Linux

---

## 🛠️ 1. Environment Setup

It is strongly recommended to use a virtual environment to keep dependencies isolated.

### Step 1.1: Clone & Navigate to Repository
```bash
cd Team-NARVL
```

### Step 1.2: Create Virtual Environment
```bash
python -m venv .venv
```

### Step 1.3: Activate Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  > *If you encounter `execution of scripts is disabled on this system`, run:*
  > ```powershell
  > Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  > .venv\Scripts\Activate.ps1
  > ```

- **Windows (Command Prompt / CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

Once activated, your terminal prompt will be prefixed with `(.venv)`.

---

## 📦 2. Install Dependencies

Install all core, UI, validation, and storage dependencies:

```bash
# 1. Install all dependencies from requirements.txt
python -m pip install -r requirements.txt

# 2. Install the NARVL package in editable mode
python -m pip install -e .
```

---

## 🚀 3. Running the Project

NARVL can be executed in three primary modes:

### Mode 1: CleanPilot Interactive Streamlit Web UI

Launch the 8-screen autonomous data cleaning interface:

```bash
# Using the NARVL CLI wrapper:
narvl ui

# OR directly with Streamlit:
python -m streamlit run narvl/ui/app.py
```

- **Default URL**: [http://localhost:8501](http://localhost:8501)
- **Features**: Drag-and-drop CSV/Parquet uploads, live profiling, ONNX semantic typing, approximate FD mining, 4D loss cards, and Delta Lake ACID rollback.

---

### Mode 2: Autonomous CLI Cleaning Pipeline

Execute end-to-end cleaning autonomously on tabular datasets:

```bash
narvl clean data/demo_enterprise.csv -o data/cleaned_output.csv --delta-dir ./delta_store
```

#### Output Artifacts:
1. `cleaned_output.csv` — Cleaned tabular asset.
2. `narvl_provenance_report.json` — HMAC-SHA256 signed audit report.
3. `narvl_provenance_report.html` — Interactive executive certificate.
4. `delta_store/_delta_log/` — ACID commit logs enabling 100% bitwise parity rollback.

---

### Mode 3: Enterprise REST API Server

Run the production FastAPI backend:

```bash
# Using the NARVL CLI launcher:
narvl serve --port 8000

# OR using Uvicorn directly:
python -m uvicorn narvl.api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Health Endpoint**: `http://localhost:8000/healthz`
- **Swagger Interactive API Docs**: `http://localhost:8000/docs`
- **Prometheus Metrics**: `http://localhost:8000/metrics`

---

## 🧪 4. Running Tests

To verify your installation and run the automated test suite:

```bash
pytest
```

---

## 🐳 5. Running with Docker (Alternative)

NARVL includes a multi-stage, air-gapped production Docker setup.

### Build Docker Image:
```bash
docker build -t narvl:latest .
```

### Run Web UI via Docker:
```bash
docker run -p 8501:8501 -p 8000:8000 narvl:latest
```
Access the UI at `http://localhost:8501`.

---

## ❓ 6. Common Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `streamlit : The term 'streamlit' is not recognized` | Virtual environment is not activated, or `streamlit` wasn't installed. | Activate `.venv` (`.venv\Scripts\Activate.ps1`) and run via `python -m streamlit run narvl/ui/app.py`. |
| `running scripts is disabled on this system` | Windows PowerShell execution policy restriction. | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell session. |
| `Address already in use (Port 8501)` | Another instance is already bound to port 8501. | Run on a different port: `narvl ui --port 8502` or `python -m streamlit run narvl/ui/app.py --server.port 8502`. |
| `No module named narvl` | Package not installed in editable mode. | Run `python -m pip install -e .` from the project root while `.venv` is active. |

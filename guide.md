# NARVL Processing and Library Guide

## 1. What NARVL Does

NARVL is an offline-first data-cleaning platform. It accepts tabular files or API records, inspects them for structural and semantic problems, creates a constrained cleaning plan, estimates the risk of that plan, executes only approved transformations, validates the result, and emits reversible audit artifacts.

The core design principle is that raw data should be handled by deterministic local components. The planner receives compact metadata, semantic classifications, and functional dependencies rather than unrestricted raw records. Cleaning is therefore explainable, testable, and reversible.

The main runtime paths are:

- CLI: `narvl clean`, `narvl ui`, and `narvl serve` in `narvl/cli.py`.
- Streamlit UI: the eight-screen CleanPilot application in `narvl/ui/app.py`.
- REST API: the FastAPI application in `narvl/api/main.py`.

## 2. Dependency Groups

The dependency groups are defined in `pyproject.toml`. The project also has a `requirements.txt` file for environment installation; use the grouped `pyproject.toml` extras when possible because they express which features are optional.

### 2.1 Core data and text libraries

| Library | Purpose | Main NARVL usage |
| --- | --- | --- |
| `polars` | High-performance DataFrame engine. | Reads normalized data, calculates profiles, performs vectorized transformations, filters rows, and converts data to Arrow for Delta Lake. |
| `duckdb` | Embedded analytical SQL engine. | Provides the profiling layer with a fast local analytical engine. |
| `numpy` | Numerical arrays and vectorized math. | Creates feature matrices, calculates cosine drift, statistical scores, and LightGBM inputs. |
| `scipy` | Scientific and statistical algorithms. | Supplies `scipy.stats.wasserstein_distance` for distribution-shift measurement. |
| `rapidfuzz` | Fast fuzzy-string comparison. | Clusters spelling variants and detects approximate functional dependencies such as `Telengana` versus `Telangana`. |
| `charset-normalizer` | Character-encoding detection. | Determines how incoming byte streams should be decoded before parsing. |
| `ftfy` | Unicode and mojibake repair. | Repairs broken text after decoding, including malformed sequences such as `CafÃ©`. |
| `tiktoken` | Token counting and tokenization. | Measures the compact profile summary so planner input stays below the configured token budget. |

### 2.2 Semantic and model libraries

| Library | Purpose | Main NARVL usage |
| --- | --- | --- |
| `onnx` | Builds and serializes ONNX graphs. | Generates the small calibrated semantic-typing model when the cached model is absent. |
| `onnxruntime` | Executes ONNX models. | Runs semantic classification for columns such as City, State, Email, Currency, PostalCode, and Timestamp. |
| `huggingface-hub` | Resolves and manages model-cache resources. | Supports the multi-tier model resolution strategy used by the model loader. |
| `llama-cpp-python` | Runs local GGUF language models through llama.cpp. | Optionally loads a local Qwen-compatible model for constrained plan generation. If unavailable, NARVL uses its deterministic planner fallback. |
| `lightgbm` | Gradient-boosted tree modeling. | Trains small raw-versus-cleaned proxy models to detect predictive utility loss. |

The ONNX and ONNX Runtime roles are different: `onnx` creates the model file, while `onnxruntime` executes it. Both are required by the current semantic-typing path.

### 2.3 Storage and validation libraries

| Library | Purpose | Main NARVL usage |
| --- | --- | --- |
| `deltalake` | Python bindings for the Delta Lake `delta-rs` engine. | Commits each execution version, stores transaction history, supports time travel, and enables rollback. |
| `pandera` | DataFrame schema validation. | Builds blocking schemas and checks column types, null rules, ranges, and other generated constraints. |
| `great-expectations` | Declarative data-quality validation. | Builds and runs an additional expectation suite for the cleaned result. |

### 2.4 User interfaces and service libraries

| Library | Purpose | Main NARVL usage |
| --- | --- | --- |
| `streamlit` | Interactive Python web application framework. | Renders the CleanPilot upload, profile, reasoning, plan, loss, execution, validation, and provenance screens. |
| `fastapi` | ASGI web API framework. | Exposes health, authentication, cleaning, provenance, metrics, and rollback endpoints. |
| `uvicorn` | ASGI application server. | Runs the FastAPI application. |
| `pydantic` | Typed request and response validation. | Defines FastAPI payload models and validates incoming API data. It is also used by FastAPI and the validation ecosystem. |
| `pandas` | General-purpose tabular library. | Used by validation integrations and UI conversion paths such as `to_pandas()` for display. |
| `altair` | Declarative charting library. | Used by the current Streamlit UI for visual chart components. Streamlit supplies it as part of its normal visualization stack. |

### 2.5 Development and packaging libraries

| Library | Purpose |
| --- | --- |
| `pytest` | Runs the automated test suite. |
| `build` | Builds source and wheel distributions. |
| `setuptools` | Provides the Python package build backend and package discovery. |
| `wheel` | Provides wheel packaging support. |

### 2.6 Python standard library

NARVL also relies heavily on the Python standard library, so these do not need to be installed separately:

- `pathlib`, `os`, `shutil`, `tempfile`, and `importlib.resources` manage paths, caches, temporary files, and model resources.
- `json` parses plans, input records, reports, and API payloads.
- `re` supplies regex-based validation and normalization.
- `dataclasses` defines structured reports and results.
- `typing` supplies public type annotations.
- `logging` records operational and safety events.
- `hashlib`, `hmac`, `secrets`, and `base64` support signed provenance and API authentication.
- `argparse`, `subprocess`, and `sys` support the CLI.
- `datetime`, `time`, `uuid`, and `io` support timestamps, measurements, identifiers, and in-memory file handling.

## 3. Processing Stages

The platform uses L0 through L6 labels. L2.5 contains two parallel analysis branches, and the L5 label is used both for planning governance and post-execution validation.

```text
Input file or API records
        |
        v
L0 Streaming Shield
        |
        v
L1 Normalizer
        |
        +--> L2 Profiler
        |
        +--> L2.5 Semantic Typer
        |
        +--> L2.5 Functional Dependency Miner
                         |
                         v
                 L3 Plan Generator
                         |
                         v
                 L5 Confidence Gate
                         |
                         v
                 L4 Loss and Utility Gate
                         |
                         v
                 L6 Reversible Executor
                    /             \
                   v               v
          L5 Validation      Provenance
                               and rollback
```

### Stage L0: Streaming adversarial shield

Implementation: `narvl/core/shield.py`, class `StreamingShield`.

The shield is the first boundary for file input. It works in binary chunks rather than loading an uncontrolled file into memory.

1. Reads an initial sample, normally in 64 KB chunks.
2. Enforces the cumulative default quota of 500 MB.
3. Sniffs the format from the extension and byte signatures.
4. Detects the character encoding using `charset-normalizer`.
5. Removes null bytes and repairs non-ASCII text with `ftfy`.
6. Detects the dominant delimiter for delimited text.
7. Checks row structure and routes malformed or suspicious rows to `quarantine.log`.
8. Produces a `ShieldResult` containing paths, row counts, encoding, delimiter, and quarantine statistics.

Parquet is treated as a binary format and copied after quota validation. JSON receives additional syntax-recovery handling, including malformed object lists and NDJSON-like content.

Important functions and types:

- `detect_encoding`: identifies the best available text encoding.
- `detect_delimiter`: scores comma, tab, semicolon, and pipe delimiters.
- `sniff_format`: identifies CSV, TSV, JSON, NDJSON, or Parquet.
- `inspect_and_clean`: performs the full streaming inspection and sanitization.
- `sanitize_file`: convenience entry point used by the CLI and UI.
- `QuotaExceededError`: stops processing when the byte ceiling is exceeded.

### Stage L1: format normalization

Implementation: `narvl/core/normalizer.py`, class `DatasetNormalizer`.

The normalizer converts supported input formats into a consistent `polars.DataFrame`:

- CSV
- TSV and TAB
- Parquet
- JSON arrays and objects
- NDJSON and JSONL

`detect_format` uses the extension and content sniffing. Polars is the fast path for tabular reads. Python's `json` parser is used as a fallback for malformed or nested JSON. `_postprocess_dataframe` applies consistent cleanup after loading. The result can be represented by `NormalizedDataset`, which combines the DataFrame with source format, source path, and shield metadata.

### Stage L2: profiling and anomaly scan

Implementation: `narvl/core/profiler.py`, class `DatasetProfiler` and alias `FastProfiler`.

The profiler creates a compact, deterministic summary rather than passing the entire dataset to the planner. It calculates:

- row and column counts;
- duplicate rows and duplicate percentage;
- repeated tuples and event-log detection;
- likely identifier or primary-key columns;
- data types, null percentages, and cardinality;
- numeric minimum, maximum, mean, standard deviation, and quartiles;
- zero-variance columns;
- representative string samples;
- email, phone, ISO date, and postal-code pattern ratios.

`tiktoken` measures the serialized summary size. The result is stored in `DatasetProfile`, which contains `summary_dict`, compact `summary_json`, `token_count`, and profiling duration.

`get_columns_with_anomalies` selects columns with nulls, zero variance, or recognized age-boundary violations. The CleanPilot UI uses this method for its “Select Columns with Issues Only” workflow.

### Stage L2.5A: semantic typing

Implementation: `narvl/core/semantic_typer.py`, class `SemanticTyper`.

The semantic typer classifies columns into enterprise-oriented categories:

- `City`
- `PostalCode`
- `State`
- `Email`
- `Currency`
- `Timestamp`
- `Other` or `Unknown`

For each column, `extract_features` creates ten normalized features from a sample of values. These include character ratios, email and postal matches, currency symbols, date delimiters, city/state lookup matches, and normalized value length.

`ensure_onnx_model` creates the small calibrated ONNX graph when the model is not cached. `onnxruntime` then evaluates the model. Predictions below the confidence threshold, or predictions of `Other`, become `Unknown` rather than being treated as reliable semantic types.

The model loader resolves model files through configured local paths and cache locations. Heavy model binaries are intentionally excluded from the Python wheel.

### Stage L2.5B: functional dependency mining

Implementation: `narvl/core/fd_miner.py`, class `FunctionalDependencyMiner`.

The FD miner discovers relationships of the form `X -> Y`, where a determinant column predicts a dependent column.

1. It first checks exact dependencies using grouped unique counts.
2. For text columns that are not exact, it groups observed values and clusters similar variants with `rapidfuzz`.
3. It selects a dominant canonical value and records non-identity mappings.
4. It calculates confidence from the dominant clustered counts.
5. It returns `FunctionalDependency` objects containing determinant, dependent, exactness, confidence, mappings, and sample violations.

This allows the planner to turn small spelling variants into explicit canonicalization steps while retaining confidence information.

### Stage L3: constrained cleaning-plan generation

Implementation: `narvl/engine/planner.py`, class `SLMPlanner`; schema rules in `narvl/engine/grammars.py`.

The planner combines:

- the compact profiler summary;
- semantic classifications;
- discovered functional dependencies;
- selected target columns;
- representative raw DataFrame values when available.

If `llama-cpp-python` and a local GGUF model are available, it can ask the local model for a plan. The plan is constrained by the GBNF grammar so each step has the expected fields, including step ID, target column, action, parameters, confidence, justification, loss potential, and test criterion.

If the local model is unavailable, `_generate_deterministic_plan` creates the plan locally using the same domain rules. This fallback handles duplicate rows, city/state/phone standardization, nulls, age bounds, financial bounds, and email validation.

`validate_cleaning_step` and `validate_cleaning_plan` reject invalid actions, missing keys, malformed parameters, or unsupported plans before execution.

### L5: confidence governance

The planner routes each valid step using `CONFIDENCE_GATE_THRESHOLD`, currently 0.85:

- Confidence greater than or equal to 0.85 goes to `auto_batch`.
- Confidence below 0.85 goes to `human_review_queue`.

This gate controls whether a recommendation can be applied automatically or requires human review. It is separate from the later information-loss gate, which evaluates the proposed result itself.

### L4: information-loss and utility safety gate

Implementation: `narvl/core/loss.py`, class `LossEstimator`.

The loss estimator compares raw and candidate-cleaned data across four dimensions:

1. **Volumetric loss**: percentage of rows removed.
2. **Statistical W1**: standardized Wasserstein distance for numeric distributions.
3. **Categorical Jaccard loss**: change in categorical value sets.
4. **Semantic cosine drift**: change in deterministic text feature centroids.

It also runs a speculative LightGBM utility check. A small classifier or regressor is trained on raw and cleaned samples, and both are evaluated on the raw reference distribution. A negative cleaned-minus-raw score indicates predictive utility loss.

`LossImpactReport` stores all measurements, threshold results, blocking reasons, and a mitigation recommendation. The safety gate blocks plans when thresholds are exceeded, including excessive row loss, distribution drift, semantic drift, or negative utility delta.

The estimator exposes these important methods:

- `calculate_volumetric_loss`
- `calculate_statistical_w1`
- `calculate_categorical_jaccard`
- `calculate_semantic_drift`
- `evaluate_speculative_utility`
- `evaluate_loss_and_safety`
- `assess`, a convenience alias for the full assessment

### L6: reversible execution

Implementation: `narvl/core/executor.py`, class `ReversibleExecutor`.

The executor applies approved plan steps with Polars expressions. Supported operations include:

- exact deduplication;
- canonical value standardization;
- numeric bound clamping;
- median or mode null imputation;
- regex replacement, lowercasing, and invalid-value nullification.

Target-column selection is enforced before execution. The executor can run in dry-run mode for loss simulation without writing a Delta commit.

For committed execution:

1. The raw DataFrame is initialized as Delta version 0.
2. Each step can create a new Delta version.
3. Metadata records the operation, target, row count, and column count.
4. `get_commit_history` exposes the transaction history.
5. `rollback_to_version` restores a previous version using Delta Lake time travel.

`resolve_nulls` uses median values for numeric columns and mode values for categorical or Boolean columns. It avoids fabricating values for identifiers, primary keys, and email columns; unresolved nulls can instead cause affected rows to be removed.

### L5: dual validation

Implementation: `narvl/core/test_gen.py`, class `DualTestSynthesizer`.

The synthesizer converts the cleaning plan into two validation systems:

- A Pandera DataFrame schema for structural and data constraints.
- A Great Expectations suite for expectation-style quality checks.

Validation checks can cover non-null requirements, type compatibility, numeric ranges, email formats, uniqueness, and plan-specific criteria. The result is represented by `DualValidationResult`, including pass/fail status, details, and blocking failures.

Execution should only be considered complete when the validation result is acceptable, even if the loss gate passed.

### Provenance and audit reporting

Implementation: `narvl/core/provenance.py`, class `ProvenanceReporter`.

The provenance layer creates an auditable record of the operation:

- raw and cleaned dataset hashes;
- plan steps and execution versions;
- loss assessment;
- validation results;
- timestamps and operational metadata.

`compute_df_sha256` hashes a DataFrame representation. The reporter signs manifest data with HMAC-SHA256 and emits both machine-readable JSON and an executive HTML report.

### CLI, UI, and API orchestration

The core stages can be reached through three interfaces.

#### CLI

`narvl/cli.py` provides `run_clean_pipeline` and `main`. The CLI coordinates shield, normalization, profiling, semantic typing, FD mining, planning, loss assessment, execution, validation, and provenance generation.

#### Streamlit CleanPilot UI

`narvl/ui/app.py` exposes the same workflow as eight interactive screens:

1. Ingestion Shield and upload.
2. Dataset profile and anomaly radar.
3. Semantic reasoning and functional dependencies.
4. Interactive cleaning-plan builder.
5. 4D loss simulator.
6. Reversible execution stepper.
7. Dual validation results.
8. Before/after comparison and provenance downloads.

The UI uses `session_state` to carry the raw DataFrame, profile, semantic results, selected columns, plan steps, execution results, validation results, and provenance artifacts between screens.

#### REST API

`narvl/api/main.py` provides:

- `/healthz` for health checks;
- `/metrics` for Prometheus-style observability;
- `/api/v1/auth/token` for token creation;
- `/api/v1/clean` for cleaning requests;
- `/api/v1/provenance` for audit report retrieval;
- `/api/v1/rollback` for administrative Delta version rollback.

FastAPI and Pydantic validate request payloads. HMAC-backed JWT-style tokens and role checks protect operator, auditor, and administrator actions.

## 4. Model Resolution and Offline Behavior

NARVL is designed to run without sending raw data to an external model service. Model resolution follows the local/cache strategy implemented by `narvl/engine/model_loader.py`:

1. A path supplied by the `NARVL_MODEL_DIR` environment variable.
2. The user cache under `~/.cache/narvl/models/`.
3. A locally available Hugging Face cache.
4. The container path `/app/models/`.

The SLM planner can operate without `llama-cpp-python` by using its deterministic fallback. The semantic typer can generate its compact ONNX model locally, but it still needs both `onnx` and `onnxruntime` installed.

## 5. Installation and Running

From the repository root, configure the virtual environment and install the package with the desired feature groups:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .[ml,ui]
```

For local SLM support, include the `slm` extra. On Windows, `llama-cpp-python` may require Visual Studio C++ Build Tools if a compatible prebuilt wheel is not available:

```powershell
python -m pip install .[ml,ui,slm]
```

Run the Streamlit UI from the repository root with the virtual-environment interpreter:

```powershell
.\.venv\Scripts\python.exe -m streamlit run narvl\ui\app.py
```

Run the CLI:

```powershell
.\.venv\Scripts\narvl.exe clean input.csv -o cleaned.csv --delta-dir .\delta_store
```

Run the REST API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn narvl.api.main:app --host 0.0.0.0 --port 8000
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

## 6. Output Artifacts

A completed cleaning run can produce:

- the cleaned CSV or other requested tabular output;
- a Delta Lake directory with `_delta_log` transaction history;
- a JSON provenance report;
- an HTML provenance report;
- a quarantine log for malformed or isolated input rows;
- validation results and blocking failure details.

The combination of loss assessment, dual validation, Delta versioning, and signed provenance is what turns a transformation into an auditable cleaning workflow rather than an untracked DataFrame mutation.

# Data Flow Diagrams
## NARVL Offline Autonomous Agentic Data Cleaning Platform

**Version:** 1.0
**Notation:** Mermaid flowcharts use external entities, processes, data stores, and labeled data flows. Processes are numbered to align with the implementation stages in `guide.md` and `README.md`.

## 1. DFD Scope and Conventions

- **External entities** are users, client applications, input files, and local model assets.
- **Processes** are NARVL services or stage components.
- **Data stores** are temporary files, Delta tables, caches, session state, and reports.
- **Data flows** are records, metadata, plans, validation results, or artifacts.
- The main file pipeline is bounded by L0 through L6. API requests enter through the same core processing services after records are converted to a Polars DataFrame.

## 2. Context Diagram

At context level, NARVL is one data-cleaning system surrounded by users, client applications, local files, optional model assets, and operational consumers.

```mermaid
flowchart LR
    User[Data Engineer or Steward]
    APIClient[REST API Client]
    Input[Raw Dataset File]
    ModelAssets[Local or Cached Model Assets]
    NARVL((NARVL Cleaning Platform))
    Cleaned[Cleaned Dataset]
    Audit[Signed Audit Reports]
    Ops[Auditor or Administrator]
    Metrics[Monitoring Client]

    User -->|Upload, select columns, approve or review| NARVL
    APIClient -->|Authenticated records and options| NARVL
    Input -->|CSV, TSV, Parquet, JSON, NDJSON| NARVL
    ModelAssets -->|ONNX or GGUF model files| NARVL
    NARVL -->|Cleaned CSV or Parquet| Cleaned
    NARVL -->|JSON/HTML provenance and quarantine details| Audit
    Ops -->|Retrieve reports or request rollback| NARVL
    NARVL -->|Health and Prometheus-style metrics| Metrics
```

## 3. Level-1 End-to-End DFD

```mermaid
flowchart TD
    E1[User or API Client]
    E2[Input File]
    E3[Local Model Cache]

    P0[0. Interface and Job Intake]
    P1[1. L0 Streaming Shield]
    P2[2. L1 Format Normalizer]
    P3[3. L2 Profiler]
    P4[4. L2.5 Semantic Typer]
    P5[5. L2.5 FD Miner]
    P6[6. L3 Planner and Grammar Validator]
    P7[7. L5 Confidence Governance]
    P8[8. L4 Loss and Utility Gate]
    P9[9. L6 Delta Executor]
    P10[10. L5 Dual Validator]
    P11[11. Provenance Reporter]

    D1[(Temporary Sanitized Input)]
    D2[(Profile and Plan State)]
    D3[(Delta Lake Table and _delta_log)]
    D4[(Model Cache)]
    D5[(Quarantine Log)]
    D6[(JSON/HTML Reports)]

    E1 -->|Job options and approval| P0
    E2 -->|Raw bytes or records| P0
    E3 -->|ONNX/GGUF assets| D4
    P0 --> P1
    P1 -->|Sanitized bytes| D1
    P1 -->|Malformed rows| D5
    D1 --> P2
    P2 -->|Normalized Polars DataFrame| P3
    P2 -->|Normalized Polars DataFrame| P4
    P2 -->|Normalized Polars DataFrame| P5
    P3 -->|Compact profile summary| D2
    P3 --> P6
    P4 -->|Semantic types and confidence| P6
    P5 -->|FDs and canonical mappings| P6
    D4 --> P4
    D4 --> P6
    P6 -->|Grammar-valid plan| P7
    P7 -->|Auto batch and review queue| P8
    P7 -->|Human review required| E1
    P8 -->|Candidate DataFrame and loss report| D2
    P8 -->|Blocked plan and reasons| E1
    P8 -->|Safe candidate| P9
    P9 -->|Versioned cleaned data| D3
    D3 --> P10
    P10 -->|Validation result and quarantined failures| D2
    P9 --> P11
    P10 --> P11
    D2 --> P11
    P11 --> D6
    P9 -->|Cleaned output| E1
    D6 -->|Signed JSON/HTML| E1
```

## 4. Level-2 File Ingestion and Normalization DFD

```mermaid
flowchart LR
    F[Input File]
    S1[Read bounded byte chunk]
    S2[Detect format and encoding]
    S3[Repair Unicode and remove null bytes]
    S4[Validate rows and recover JSON syntax]
    S5[Detect delimiter and quarantine bad rows]
    Q[(quarantine.log)]
    C[(Sanitized file)]
    N1[Select Polars reader]
    N2[Fallback Python JSON/NDJSON reader]
    N3[Post-process schema]
    DF[(Normalized Polars DataFrame)]

    F --> S1 --> S2 --> S3 --> S4 --> S5
    S5 -->|Valid sanitized stream| C
    S5 -->|Ragged, malformed, or suspicious rows| Q
    C --> N1
    C --> N2
    N1 --> N3
    N2 --> N3
    N3 --> DF
```

### Ingestion data controls

- `StreamingShield` maintains `total_bytes` and raises `QuotaExceededError` after the configured limit.
- `charset-normalizer` supplies the selected encoding; UTF-8 is the fallback.
- `ftfy` repairs non-ASCII decoded text.
- JSON recovery may convert malformed object sequences into arrays or extract valid objects with `json.JSONDecoder.raw_decode`.
- Parquet is copied after binary-size validation rather than text-decoded.

## 5. Level-2 Analysis DFD

The normalized DataFrame is read by three analysis branches. They produce metadata rather than a second copy of the entire dataset for the planner.

```mermaid
flowchart TD
    DF[(Normalized Polars DataFrame)]
    Prof[DatasetProfiler]
    Type[SemanticTyper]
    FD[FunctionalDependencyMiner]
    Profile[(DatasetProfile summary)]
    Types[(SemanticClassification map)]
    FDs[(FunctionalDependency list)]
    Cache[(ONNX model cache)]
    Token[tiktoken token count]

    DF --> Prof
    DF --> Type
    DF --> FD
    Prof --> Profile
    Prof --> Token
    Token --> Profile
    Cache --> Type
    Type --> Types
    FD --> FDs
```

### Analysis flows

1. `DatasetProfiler.profile` calculates dataset and column metadata, serializes the compact summary, and records token count and duration.
2. `SemanticTyper.infer_types` samples each column, extracts ten features, invokes ONNX Runtime, and returns semantic type/confidence records.
3. `FunctionalDependencyMiner.mine` checks exact mappings and clusters text variants with RapidFuzz before producing confidence-scored dependency records.

## 6. Level-2 Planning and Governance DFD

```mermaid
flowchart LR
    Profile[(Profile summary)]
    Types[(Semantic types)]
    FDs[(Functional dependencies)]
    Selected[Selected target columns]
    Raw[Representative raw DataFrame]
    Model[(Optional local GGUF model)]
    Planner[SLMPlanner]
    Grammar[GBNF grammar validation]
    Plan[(Cleaning plan JSON)]
    Gate[L5 confidence governance]
    Auto[(Auto-batch steps)]
    Review[(Human-review steps)]
    User[User or API caller]

    Profile --> Planner
    Types --> Planner
    FDs --> Planner
    Selected --> Planner
    Raw --> Planner
    Model --> Planner
    Planner --> Grammar --> Plan --> Gate
    Gate -->|confidence >= 0.85| Auto
    Gate -->|confidence < 0.85| Review
    Review --> User
```

The planner may use the local SLM, but the deterministic fallback is the authoritative availability path when model loading fails. Grammar validation is applied to either source of plan steps.

## 7. Level-2 Safety, Execution, and Validation DFD

```mermaid
flowchart TD
    Raw[(Raw normalized DataFrame)]
    Plan[(Approved plan steps)]
    Sim[Dry-run executor]
    Candidate[(Candidate cleaned DataFrame)]
    Loss[LossEstimator]
    W1[Wasserstein W1]
    Jaccard[Categorical Jaccard]
    Cosine[Semantic cosine drift]
    Vol[Volumetric loss]
    LGBM[LightGBM utility proxy]
    Verdict{Safety gate}
    Commit[ReversibleExecutor commit mode]
    Delta[(Delta Lake versions)]
    Validate[DualTestSynthesizer]
    PSchema[Pandera schema]
    GESuite[Great Expectations suite]
    Valid[(Validation result)]
    Clean[(Cleaned DataFrame)]
    Quarantine[(Validation quarantine)]
    User[User or API client]

    Raw --> Sim
    Plan --> Sim
    Sim --> Candidate
    Raw --> Loss
    Candidate --> Loss
    Loss --> Vol
    Loss --> W1
    Loss --> Jaccard
    Loss --> Cosine
    Loss --> LGBM
    Vol --> Verdict
    W1 --> Verdict
    Jaccard --> Verdict
    Cosine --> Verdict
    LGBM --> Verdict
    Verdict -->|Blocked| User
    Verdict -->|Safe or explicitly approved| Commit
    Plan --> Commit
    Raw --> Commit
    Commit --> Delta
    Delta --> Clean
    Clean --> Validate
    Plan --> Validate
    Validate --> PSchema
    Validate --> GESuite
    PSchema --> Valid
    GESuite --> Valid
    Valid --> Quarantine
    Valid --> User
```

## 8. Level-2 API DFD

```mermaid
sequenceDiagram
    participant C as API Client
    participant A as FastAPI
    participant Auth as HMAC/RBAC
    participant Core as NARVL Core Pipeline
    participant Delta as Delta Lake
    participant Report as Provenance Reporter

    C->>A: POST /api/v1/auth/token
    A-->>C: Bearer token
    C->>A: POST /api/v1/clean + records + options
    A->>Auth: Verify token and operator/admin role
    Auth-->>A: Authorized user
    A->>Core: Convert records to Polars DataFrame
    Core->>Core: Profile, type, mine FDs, plan, simulate, assess
    alt Safety gate blocked and auto_approve=false
        Core-->>A: Blocking reasons and mitigation
        A-->>C: HTTP 422 SafetyGateBlocked
    else Safe or explicitly approved
        Core->>Delta: Commit plan versions
        Core->>Core: Validate with Pandera and GE
        Core->>Report: Sign provenance manifest
        Report-->>A: Report dictionary and signature
        A-->>C: Cleaned records, loss summary, version, validation
    end
    C->>A: GET /api/v1/provenance
    A->>Auth: Verify operator/admin/auditor role
    A-->>C: Latest signed report
    C->>A: POST /api/v1/rollback
    A->>Auth: Verify admin role
    A->>Delta: Load requested historical version
    Delta-->>A: Restored DataFrame
    A-->>C: Rollback result
```

## 9. Data Stores and Lifetimes

| Store | Contents | Lifetime |
| --- | --- | --- |
| Temporary sanitized input | Shielded copy of the source file. | Job/session; normally local temporary storage. |
| `quarantine.log` | Malformed, ragged, or rejected input records and reasons. | Job/report directory. |
| Profile/session state | Profile summary, semantic types, FDs, selected columns, plan, loss result, and validation result. | Streamlit session or latest API process session. |
| Model cache | ONNX semantic model and optional GGUF SLM model. | User cache or configured model directory. |
| Delta table | Versioned raw and cleaned snapshots plus `_delta_log`. | Job storage; retained for rollback/audit. |
| Provenance reports | Signed JSON manifest and HTML report. | Report directory or API latest-session state. |
| Cleaned output | CSV or Parquet asset. | User-selected output location. |

## 10. Error and Control Flows

```mermaid
flowchart TD
    Start[Processing request] --> Quota{Quota exceeded?}
    Quota -->|Yes| Stop1[Abort with QuotaExceededError]
    Quota -->|No| Parse{Input recoverable?}
    Parse -->|No| Quarantine[Write quarantine record]
    Parse -->|Yes| Analyze[Analyze normalized data]
    Quarantine --> Analyze
    Analyze --> Model{Model available?}
    Model -->|ONNX available| Type[Semantic type inference]
    Model -->|ONNX unavailable| ErrorType[Return dependency/model error]
    Analyze --> Plan[Build plan]
    Type --> Plan
    Plan --> Grammar{Plan valid?}
    Grammar -->|No| Stop2[Reject plan]
    Grammar -->|Yes| Loss{Safety thresholds pass?}
    Loss -->|No and no override| Stop3[Block execution]
    Loss -->|Yes or explicit override| Execute[Commit Delta versions]
    Execute --> Validation{Validation passes?}
    Validation -->|No| Filter[Quarantine/filter invalid records]
    Validation -->|Yes| Publish[Publish clean output and reports]
    Filter --> Publish
```

## 11. DFD Security Boundaries

- The ingestion boundary is the first untrusted-data boundary. Quota, format, encoding, null-byte, delimiter, and quarantine controls apply there.
- The planner boundary receives compact metadata and controlled samples rather than an unrestricted raw-data prompt.
- The execution boundary is protected by dry-run loss assessment and the safety gate.
- The API boundary is protected by HMAC-signed Bearer tokens and role checks.
- The audit boundary includes signed provenance and immutable Delta versions.
- The current API latest-session store is process-local; a horizontally scaled deployment would require an external shared store and coordinated authorization.

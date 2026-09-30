"""
Command Line Interface for NARVL Document Intelligence Extension.
Provides `clean-document` command for end-to-end autonomous document cleaning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from narvl.document.exporters import DocumentExporter
from narvl.document.pipeline import DocumentCleaningPipeline


def run_clean_document(
    input_file: Path,
    output_dir: Optional[Path] = None,
    auto_approve: bool = False,
) -> int:
    """Run full document intelligence and cleaning workflow."""
    print("=======================================================")
    print("[*] NARVL: Unstructured Document Intelligence Engine")
    print("=======================================================")
    print(f"[*] Target document: {input_file}")

    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}", file=sys.stderr)
        return 1

    # Base directory is doc_extension
    base_dir = Path(__file__).resolve().parent.parent
    if output_dir is None:
        output_dir = base_dir / "demo_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline initialization
    print("\n[Stage 1/7] Initializing Document Pipeline & Ingesting through L0 Shield...")
    pipeline = DocumentCleaningPipeline(base_dir=base_dir, auto_approve=auto_approve)

    # Run flow
    res = pipeline.run_cleaning_flow(input_file, auto_approve=auto_approve)
    doc_id = res["doc_id"]
    print(f"  [+] Document registered: doc_id={doc_id}")
    print(f"  [+] Preserved original copy at: {res['original_path']}")

    # 2. Extraction summary
    print("\n[Stage 2/7] Extracted Document Structure & Tabular Data...")
    parsed = res["parsed_document"]
    print(f"  [+] Document Type: {parsed.file_type.value}")
    print(f"  [+] Pages: {len(parsed.pages)}, Tables: {len(parsed.tables)}, Headings: {len(parsed.structure.headings)}")

    # 3. Entity Extraction & Relations
    print("\n[Stage 3/7] Extracted Semantic Entities & Relation Graph...")
    print(f"  [+] Extracted {res['raw_entity_count']} entities across categories")
    graph = res["graph"]
    print(f"  [+] Detected {len(graph.edges)} entity relationships; {len(graph.clusters)} primary clusters formed")

    # 4. Data Quality Detection
    print("\n[Stage 4/7] Detected Data Quality Issues (6 Issue Classes)...")
    issues = res["quality_issues"]
    print(f"  [+] Discovered {len(issues)} data quality anomalies:")
    for iss in issues[:8]:
        print(f"      - [{iss['severity'].upper()}] {iss['type']} on {iss['field']}: {iss['evidence']}")
    if len(issues) > 8:
        print(f"      ... and {len(issues) - 8} more issues.")

    # 5. Cleaning Recommendations & Governance
    print("\n[Stage 5/7] Formulating Cleaning Recommendations & L4/L5 Governance Gate...")
    recs = res["recommendations"]
    approved = res["approved_recommendations"]
    print(f"  [+] Generated {len(recs)} cleaning recommendations")
    print(f"  [+] Approved: {len(approved)} | Requires Human Sign-off: {len(recs) - len(approved)}")
    for r in recs[:6]:
        status_tag = "[AUTO-APPROVED]" if r["issue_id"] in [a["issue_id"] for a in approved] else "[BLOCKED/REVIEW]"
        print(f"      {status_tag} {r['operation_type']} (confidence: {r['confidence']:.2f}, risk: {r['risk_level']}): {r['explanation']}")

    # 6. Reversible Delta Lake Versioning
    print("\n[Stage 6/7] Committing Immutable Reversible Delta Lake Versions...")
    versions = res["versions"]
    for v in versions:
        print(f"  [+] Committed Delta Version {v['version_id']}: '{v['operation']}'")

    # 7. Validation & Exports
    print("\n[Stage 7/7] Executing Automated Validation Suite & Exporting Artifacts...")
    val = res["validation_report"]
    print(f"  [+] Validation Suite Passed: {val['passed']}")
    for chk in val["checks"]:
        icon = "PASS" if chk["passed"] else "FAIL"
        print(f"      [{icon}] {chk['name']} ({chk['target_field']}): {chk['details']}")

    exporter = DocumentExporter(output_dir=output_dir)
    exported_files = exporter.export_all(res)

    print("\n=======================================================")
    print("[SUCCESS] Autonomous Document Cleaning Completed!")
    print("Exported Deliverables:")
    for k, p in exported_files.items():
        print(f"  - {k.upper():<16}: {p}")
    print("=======================================================\n")
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="clean-document",
        description="NARVL Autonomous Document Intelligence & Reversible Cleaning Pipeline",
    )
    parser.add_argument("document", type=Path, help="Path to messy document (TXT, DOCX, PDF)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory to store cleaned deliverables")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve medium-risk operations in governance gate")

    parsed_args = parser.parse_args(args)
    return run_clean_document(
        input_file=parsed_args.document,
        output_dir=parsed_args.output_dir,
        auto_approve=parsed_args.auto_approve,
    )


if __name__ == "__main__":
    sys.exit(main())

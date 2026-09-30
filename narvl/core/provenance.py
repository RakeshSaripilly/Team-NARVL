"""
Provenance Audit Engine for NARVL.

Emits cryptographically signed audit manifests:
- narvl_provenance_report.json (machine-readable immutable provenance)
- narvl_provenance_report.html (executive audit certificate with interactive UI)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from narvl.core.loss import LossImpactReport
from narvl.core.test_gen import DualValidationResult

SIGNING_KEY = b"narvl_enterprise_audit_signature_secret_v1"


def compute_df_sha256(df: pl.DataFrame) -> str:
    """Compute deterministic SHA-256 content hash of Polars DataFrame."""
    try:
        # Convert schema and data bytes to SHA-256
        h = hashlib.sha256()
        h.update(str(df.schema).encode("utf-8"))
        for col_name in sorted(df.columns):
            ser_bytes = df[col_name].to_string().encode("utf-8")
            h.update(ser_bytes)
        return h.hexdigest()
    except Exception:
        return hashlib.sha256(str(df.shape).encode("utf-8")).hexdigest()


class ProvenanceReporter:
    """Generates immutable cryptographic audit certificates for data cleaning runs."""

    def __init__(self, signing_key: bytes = SIGNING_KEY) -> None:
        self.signing_key = signing_key

    def create_manifest(
        self,
        raw_df: pl.DataFrame,
        cleaned_df: pl.DataFrame,
        steps: List[Dict[str, Any]],
        loss_report: LossImpactReport,
        validation_result: DualValidationResult,
        dataset_name: str = "dataset",
    ) -> Dict[str, Any]:
        """Synthesize signed provenance manifest dict."""
        timestamp = datetime.now(timezone.utc).isoformat()
        raw_hash = compute_df_sha256(raw_df)
        clean_hash = compute_df_sha256(cleaned_df)
        run_id = str(uuid.uuid4())

        payload = {
            "narvl_version": "0.1.0",
            "run_id": run_id,
            "dataset_name": dataset_name,
            "timestamp_utc": timestamp,
            "lineage": {
                "raw_shape": raw_df.shape,
                "clean_shape": cleaned_df.shape,
                "input_hash_sha256": raw_hash,
                "output_hash_sha256": clean_hash,
            },
            "pipeline_steps": steps,
            "loss_profile": {
                "volumetric_loss": loss_report.volumetric_loss,
                "max_w1_distance": loss_report.max_w1,
                "w1_per_column": loss_report.w1_per_column,
                "utility_delta": loss_report.utility_delta,
                "is_safe": loss_report.is_safe,
                "blocking_reasons": loss_report.blocking_reasons,
            },
            "validation": {
                "pandera_passed": validation_result.pandera_passed,
                "ge_passed": validation_result.ge_passed,
                "is_fully_validated": validation_result.is_fully_validated,
            },
        }

        # Cryptographic HMAC signature
        serialized = json.dumps(payload, sort_keys=True)
        sig = hmac.new(self.signing_key, serialized.encode("utf-8"), hashlib.sha256).hexdigest()
        payload["digital_signature"] = sig
        payload["signature_algorithm"] = "HMAC-SHA256"

        return payload

    def emit_json(self, manifest: Dict[str, Any], output_path: Path | str) -> Path:
        """Write manifest to JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return target

    def emit_html(self, manifest: Dict[str, Any], output_path: Path | str) -> Path:
        """Generate executive HTML provenance audit certificate."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        steps_rows = ""
        for s in manifest.get("pipeline_steps", []):
            steps_rows += f"""
            <tr>
                <td style="padding:10px; border-bottom:1px solid #2d3748; font-weight:600;">#{s.get('step_id')}</td>
                <td style="padding:10px; border-bottom:1px solid #2d3748; color:#63b3ed;">{s.get('target_column')}</td>
                <td style="padding:10px; border-bottom:1px solid #2d3748;"><span style="background:#2b6cb0; padding:2px 8px; border-radius:4px; font-size:12px;">{s.get('action')}</span></td>
                <td style="padding:10px; border-bottom:1px solid #2d3748;">{s.get('confidence', 0):.2f}</td>
                <td style="padding:10px; border-bottom:1px solid #2d3748; font-size:13px; color:#a0aec0;">{s.get('justification')}</td>
            </tr>
            """

        status_badge = '<span style="background:#22543d; color:#9ae6b4; padding:6px 14px; border-radius:20px; font-weight:bold;">PASS - AUDIT VERIFIED</span>'
        if not manifest["validation"]["is_fully_validated"]:
            status_badge = '<span style="background:#742a2a; color:#feb2b2; padding:6px 14px; border-radius:20px; font-weight:bold;">BLOCKED - INTEGRITY FAILURE</span>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NARVL Immutable Provenance Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 30px; margin-bottom: 24px; border: 1px solid #334155; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 20px; }}
        h1 {{ margin: 0; font-size: 26px; color: #38bdf8; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 20px; }}
        .meta-item {{ background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid #334155; }}
        .meta-title {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: bold; }}
        .meta-val {{ font-size: 16px; font-weight: 600; margin-top: 6px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
        th {{ text-align: left; padding: 10px; background: #0f172a; color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
        .signature-box {{ background: #090d16; border: 1px dashed #475569; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 12px; color: #38bdf8; word-break: break-all; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1>NARVL Immutable Provenance Certificate</h1>
                <p style="color:#94a3b8; margin: 4px 0 0 0; font-size: 14px;">Run ID: {manifest.get('run_id')} | Generated: {manifest.get('timestamp_utc')}</p>
            </div>
            <div>{status_badge}</div>
        </div>

        <div class="meta-grid">
            <div class="meta-item"><div class="meta-title">Dataset Name</div><div class="meta-val">{manifest.get('dataset_name')}</div></div>
            <div class="meta-item"><div class="meta-title">Raw Dimensions</div><div class="meta-val">{manifest['lineage']['raw_shape']}</div></div>
            <div class="meta-item"><div class="meta-title">Clean Dimensions</div><div class="meta-val">{manifest['lineage']['clean_shape']}</div></div>
            <div class="meta-item"><div class="meta-title">Volumetric Loss</div><div class="meta-val">{manifest['loss_profile']['volumetric_loss']:.2%}</div></div>
        </div>
    </div>

    <div class="card">
        <h2 style="font-size: 18px; margin-top:0;">Pipeline Execution Plan</h2>
        <table>
            <thead>
                <tr><th>Step</th><th>Target</th><th>Action</th><th>Confidence</th><th>Justification</th></tr>
            </thead>
            <tbody>
                {steps_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2 style="font-size: 18px; margin-top:0;">Cryptographic Provenance Verification</h2>
        <p style="font-size:13px; color:#94a3b8;">Input SHA-256: <code>{manifest['lineage']['input_hash_sha256']}</code></p>
        <p style="font-size:13px; color:#94a3b8;">Output SHA-256: <code>{manifest['lineage']['output_hash_sha256']}</code></p>
        <div class="signature-box">
            [HMAC-SHA256 DIGITAL SIGNATURE]<br>
            {manifest.get('digital_signature')}
        </div>
    </div>
</body>
</html>
"""
        target.write_text(html_content, encoding="utf-8")
        return target

    def generate_report(
        self,
        raw_df: pl.DataFrame,
        cleaned_df: pl.DataFrame,
        plan_steps: List[Dict[str, Any]],
        loss_assessment: Union[LossImpactReport, Dict[str, Any]],
        validation_result: Union[DualValidationResult, Dict[str, Any]],
        dataset_name: str = "dataset",
        output_json: Optional[Union[Path, str]] = None,
        output_html: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Any]:
        """Convenience report generator with optional file export and dictionary normalization."""
        if isinstance(loss_assessment, dict):
            loss_report = LossImpactReport(
                volumetric_loss=loss_assessment.get("volumetric_loss", 0.0),
                max_w1=loss_assessment.get("max_w1", loss_assessment.get("statistical_w1", 0.0)),
                w1_per_column=loss_assessment.get("w1_per_column", {}),
                jaccard_loss=loss_assessment.get("jaccard_loss", {}),
                cosine_drift=loss_assessment.get("cosine_drift", {}),
                utility_delta=loss_assessment.get("utility_delta", loss_assessment.get("predictive_utility_delta", 0.0)),
                raw_utility=loss_assessment.get("raw_utility", 0.8),
                clean_utility=loss_assessment.get("clean_utility", 0.8),
                is_safe=loss_assessment.get("is_safe", True),
                blocking_reasons=loss_assessment.get("blocking_reasons", []),
            )
        else:
            loss_report = loss_assessment

        if isinstance(validation_result, dict):
            val_res = DualValidationResult(
                pandera_passed=validation_result.get("pandera_passed", True),
                pandera_errors=validation_result.get("pandera_errors", []),
                ge_passed=validation_result.get("ge_passed", True),
                ge_summary=validation_result.get("ge_summary", {}),
                is_fully_validated=validation_result.get("is_fully_validated", True),
            )
        else:
            val_res = validation_result

        manifest = self.create_manifest(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            steps=plan_steps,
            loss_report=loss_report,
            validation_result=val_res,
            dataset_name=dataset_name,
        )

        manifest["signature"] = manifest["digital_signature"]
        manifest["pipeline_provenance"] = manifest["pipeline_steps"]

        if output_json:
            self.emit_json(manifest, output_json)
        if output_html:
            self.emit_html(manifest, output_html)

        return manifest

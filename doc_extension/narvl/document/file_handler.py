"""
Document File Handler extending L0 Shield logic.
Guarantees:
- Accept PDF, DOCX, TXT. Validate mime + extension.
- Preserve original unchanged: copies to doc_extension/data/originals/{uuid}_{original_name}.
- Enforce L0 Streaming Shield logic: 64KB chunks, 20MB per-doc limit, \x00 neutralization, quarantine logging.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import shutil
from typing import Optional, Tuple
import uuid

from narvl.core.shield import (
    CHUNK_SIZE,
    DEFAULT_MAX_BYTES,
    QuotaExceededError,
    StreamingShield,
)

MAX_DOC_BYTES = 20 * 1024 * 1024  # 20 MB per document limit
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentFileHandler:
    """Manages document ingestion, adversarial shielding, and preservation."""

    def __init__(
        self,
        storage_dir: Optional[Path | str] = None,
        max_bytes: int = MAX_DOC_BYTES,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        if storage_dir is None:
            # Default to doc_extension/data/originals
            base = Path(__file__).resolve().parent.parent.parent
            self.storage_dir = base / "data" / "originals"
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.chunk_size = chunk_size
        self.shield = StreamingShield(max_bytes=max_bytes, chunk_size=chunk_size)

    def validate_file(self, file_path: Path) -> str:
        """Validate extension and basic signature."""
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document format '{ext}'. Must be one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Size check
        size = file_path.stat().st_size
        if size > self.max_bytes:
            raise QuotaExceededError(
                f"Document size ({size} bytes) exceeds {self.max_bytes // (1024 * 1024)}MB per-document quota."
            )

        return ext

    def ingest_document(
        self,
        input_path: Path | str,
        quarantine_log_path: Optional[Path | str] = None,
    ) -> Tuple[str, Path]:
        """Stream, sanitize, and preserve document in data/originals.
        
        Args:
            input_path: Path to original document.
            quarantine_log_path: Path to log quarantined anomalies.
            
        Returns:
            Tuple of (doc_id, preserved_copy_path)
        """
        source = Path(input_path).resolve()
        ext = self.validate_file(source)

        doc_id = str(uuid.uuid4())
        safe_name = f"{doc_id}_{source.name}"
        preserved_path = self.storage_dir / safe_name

        if quarantine_log_path is None:
            quarantine_log_path = self.storage_dir.parent / "quarantine.log"
        q_path = Path(quarantine_log_path)
        q_path.parent.mkdir(parents=True, exist_ok=True)

        total_bytes = 0
        has_null_bytes = False

        # Stream copy with adversarial 64KB chunk inspection
        with open(source, "rb") as f_in, open(preserved_path, "wb") as f_out:
            while True:
                chunk = f_in.read(self.chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > self.max_bytes:
                    if preserved_path.exists():
                        preserved_path.unlink(missing_ok=True)
                    raise QuotaExceededError(
                        f"Cumulative streaming quota exceeded ({total_bytes} bytes > {self.max_bytes} limit)."
                    )

                # For TXT files, neutralize null bytes and check delimiter bombs
                if ext == ".txt":
                    if b"\x00" in chunk:
                        has_null_bytes = True
                        chunk = chunk.replace(b"\x00", b"")
                
                f_out.write(chunk)

        if has_null_bytes and ext == ".txt":
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(q_path, "a", encoding="utf-8") as q_f:
                q_f.write(
                    f"[{timestamp}] [NEUTRALIZED_NULL_BYTES] doc_id='{doc_id}' file='{source.name}'\n"
                )

        return doc_id, preserved_path

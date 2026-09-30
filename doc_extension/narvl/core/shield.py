"""
L0: Streaming Adversarial Shield for NARVL.

Features:
- 64KB chunk binary streaming with 500MB cumulative ceiling (QuotaExceededError)
- High-speed encoding detection via charset-normalizer
- Fast BOM / Mojibake repair via ftfy.fix_text (selective ASCII bypass for 100x throughput)
- Null-byte (\x00) neutralization
- Delimiter bomb detection (statistical variance / outlier rejection)
- Fault-tolerant quarantine routing without pipeline crash
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import charset_normalizer
try:
    import ftfy
except ImportError:
    class _FtfyFallback:
        @staticmethod
        def fix_text(s: str) -> str:
            return s
    ftfy = _FtfyFallback()

CHUNK_SIZE = 64 * 1024  # 64 KB binary chunks
DEFAULT_MAX_BYTES = 500 * 1024 * 1024  # 500 MB quota ceiling


class QuotaExceededError(ValueError):
    """Raised when file streaming exceeds the cumulative byte quota limit."""


@dataclass
class ShieldResult:
    """Metadata summary emitted after streaming adversarial inspection."""
    clean_path: Path
    quarantine_path: Path
    quarantined_count: int
    valid_rows_count: int
    total_bytes_processed: int
    detected_encoding: str
    detected_delimiter: str
    detected_null_bytes: bool


class StreamingShield:
    """L0 Streaming Adversarial Shield."""

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        chunk_size: int = CHUNK_SIZE,
        delimiter_variance_threshold: float = 1.0,
    ) -> None:
        self.max_bytes = max_bytes
        self.chunk_size = chunk_size
        self.delimiter_variance_threshold = delimiter_variance_threshold

    def detect_encoding(self, sample_bytes: bytes) -> str:
        """Detect encoding using charset-normalizer."""
        try:
            clean_sample = sample_bytes.replace(b"\x00", b"")
            if not clean_sample:
                return "utf-8"
            results = charset_normalizer.from_bytes(clean_sample)
            best = results.best()
            if best and best.encoding:
                return best.encoding
        except Exception:
            pass
        return "utf-8"

    def detect_delimiter(self, sample_text: str) -> str:
        """Detect dominant tabular delimiter from a representative text sample."""
        candidates = [",", "\t", ";", "|"]
        lines = [line.strip() for line in sample_text.splitlines() if line.strip()][:30]
        if not lines:
            return ","

        scores: dict[str, float] = {}
        for d in candidates:
            counts = [line.count(d) for line in lines]
            if not counts or max(counts) == 0:
                scores[d] = 0.0
                continue
            mean_val = sum(counts) / len(counts)
            variance = sum((c - mean_val) ** 2 for c in counts) / len(counts)
            scores[d] = mean_val / (1.0 + variance)

        best_del = max(scores, key=lambda k: scores[k])
        return best_del if scores[best_del] > 0 else ","

    def inspect_and_clean(
        self,
        input_path: Path | str,
        output_clean_path: Optional[Path | str] = None,
        quarantine_log_path: Optional[Path | str] = None,
    ) -> ShieldResult:
        """Stream and sanitize dataset through L0 Shield.
        
        Args:
            input_path: Path to raw input file.
            output_clean_path: Target path for sanitized file.
            quarantine_log_path: Target path for quarantined malicious/ragged lines.
            
        Returns:
            ShieldResult with processing statistics.
            
        Raises:
            QuotaExceededError: If processed stream exceeds max_bytes.
            FileNotFoundError: If input file does not exist.
        """
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(f"Input file not found: {source}")

        if output_clean_path is None:
            output_clean_path = source.parent / f"{source.stem}_shielded{source.suffix}"
        clean_target = Path(output_clean_path)

        if quarantine_log_path is None:
            quarantine_log_path = source.parent / "quarantine.log"
        quarantine_target = Path(quarantine_log_path)

        # 1. Sample the first chunk for encoding and delimiter detection
        total_bytes = 0
        has_null_bytes = False
        initial_sample = bytearray()

        with open(source, "rb") as f_in:
            while len(initial_sample) < self.chunk_size:
                chunk = f_in.read(self.chunk_size)
                if not chunk:
                    break
                initial_sample.extend(chunk)

        if b"\x00" in initial_sample:
            has_null_bytes = True

        detected_encoding = self.detect_encoding(bytes(initial_sample))
        try:
            sample_text = initial_sample.replace(b"\x00", b"").decode(
                detected_encoding, errors="replace"
            )
        except Exception:
            sample_text = initial_sample.replace(b"\x00", b"").decode(
                "utf-8", errors="replace"
            )
        if not sample_text.isascii():
            sample_text = ftfy.fix_text(sample_text)
        detected_delimiter = self.detect_delimiter(sample_text)

        # 2. Estimate expected delimiter count from header and initial rows
        sample_lines = [l for l in sample_text.splitlines() if l.strip()]
        expected_delimiter_count: Optional[int] = None
        if sample_lines:
            header = sample_lines[0]
            header_delims = header.count(detected_delimiter)
            counts = [l.count(detected_delimiter) for l in sample_lines[:20]]
            if counts:
                counts_sorted = sorted(counts)
                median_count = counts_sorted[len(counts_sorted) // 2]
                expected_delimiter_count = median_count if median_count > 0 else header_delims

        # 3. Stream entire file in 64KB chunks, tracking 500MB quota
        quarantined_count = 0
        valid_rows_count = 0
        carryover = b""

        quarantine_target.parent.mkdir(parents=True, exist_ok=True)
        clean_target.parent.mkdir(parents=True, exist_ok=True)

        with open(source, "rb") as f_in, \
             open(clean_target, "w", encoding="utf-8", newline="") as f_clean, \
             open(quarantine_target, "a", encoding="utf-8") as f_quarantine:

            while True:
                chunk = f_in.read(self.chunk_size)
                if not chunk:
                    break
                
                # Immediate quota enforcement before costly parsing
                total_bytes += len(chunk)
                if total_bytes > self.max_bytes:
                    raise QuotaExceededError(
                        f"Cumulative 500MB ceiling exceeded ({total_bytes} bytes > {self.max_bytes} limit)."
                    )

                if b"\x00" in chunk:
                    has_null_bytes = True
                    chunk = chunk.replace(b"\x00", b"")

                data = carryover + chunk
                lines = data.split(b"\n")
                carryover = lines.pop()

                for raw_line_bytes in lines:
                    line_text = raw_line_bytes.decode(detected_encoding, errors="replace").rstrip("\r\n")
                    if not line_text.strip():
                        continue
                    
                    # Selective ftfy invocation for fast throughput on ASCII lines
                    line_fixed = line_text if line_text.isascii() else ftfy.fix_text(line_text)

                    # Inspect delimiter count against delimiter bomb / ragged conditions
                    del_count = line_fixed.count(detected_delimiter)
                    is_bad = False
                    reason = ""

                    if expected_delimiter_count is not None and expected_delimiter_count > 0:
                        diff = abs(del_count - expected_delimiter_count)
                        if del_count > max(expected_delimiter_count * 3, expected_delimiter_count + 15):
                            is_bad = True
                            reason = f"Delimiter Bomb: {del_count} delimiters (expected {expected_delimiter_count})"
                        elif diff >= self.delimiter_variance_threshold and valid_rows_count > 0:
                            is_bad = True
                            reason = f"Ragged row: {del_count} delimiters (expected {expected_delimiter_count})"
                    elif del_count > 100:
                        is_bad = True
                        reason = f"Extreme delimiter density: {del_count} delimiters"

                    if is_bad:
                        quarantined_count += 1
                        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        snippet = line_fixed[:250] + ("..." if len(line_fixed) > 250 else "")
                        f_quarantine.write(
                            f"[{timestamp}] [QUARANTINE] line={valid_rows_count + quarantined_count} "
                            f"reason='{reason}' data='{snippet}'\n"
                        )
                    else:
                        f_clean.write(line_fixed + "\n")
                        valid_rows_count += 1

            # Process final carryover line if any
            if carryover:
                total_bytes += len(carryover)
                if total_bytes > self.max_bytes:
                    raise QuotaExceededError(
                        f"Cumulative 500MB ceiling exceeded ({total_bytes} bytes > {self.max_bytes} limit)."
                    )
                final_bytes = carryover.replace(b"\x00", b"")
                line_text = final_bytes.decode(detected_encoding, errors="replace").rstrip("\r\n")
                if line_text.strip():
                    line_fixed = line_text if line_text.isascii() else ftfy.fix_text(line_text)
                    del_count = line_fixed.count(detected_delimiter)
                    if (
                        expected_delimiter_count is not None
                        and expected_delimiter_count > 0
                        and abs(del_count - expected_delimiter_count) >= self.delimiter_variance_threshold
                    ):
                        quarantined_count += 1
                        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        f_quarantine.write(
                            f"[{timestamp}] [QUARANTINE] line={valid_rows_count + quarantined_count} "
                            f"reason='Ragged end line' data='{line_fixed[:200]}'\n"
                        )
                    else:
                        f_clean.write(line_fixed + "\n")
                        valid_rows_count += 1

        return ShieldResult(
            clean_path=clean_target,
            quarantine_path=quarantine_target,
            quarantined_count=quarantined_count,
            valid_rows_count=valid_rows_count,
            total_bytes_processed=total_bytes,
            detected_encoding=detected_encoding,
            detected_delimiter=detected_delimiter,
            detected_null_bytes=has_null_bytes,
        )

    def sanitize_file(
        self,
        input_path: Path | str,
        output_clean_path: Optional[Path | str] = None,
        quarantine_log: Optional[Path | str] = None,
    ) -> tuple[Path, int]:
        """Convenience wrapper returning (clean_path, quarantined_count)."""
        res = self.inspect_and_clean(
            input_path=input_path,
            output_clean_path=output_clean_path,
            quarantine_log_path=quarantine_log,
        )
        return res.clean_path, res.quarantined_count

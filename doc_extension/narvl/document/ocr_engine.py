"""
Optional OCR Engine for scanned PDF documents.
Gracefully falls back if pytesseract or pdf2image are not installed.
Guarantees 100% offline operation without breaking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("narvl.document.ocr_engine")


class OCREngine:
    """Offline OCR extraction fallback engine."""

    def __init__(self) -> None:
        self.is_available = False
        self._tesseract = None
        self._pdf2image = None

        try:
            import pytesseract
            import pdf2image
            self._tesseract = pytesseract
            self._pdf2image = pdf2image
            self.is_available = True
        except ImportError:
            logger.info("OCR dependencies (pytesseract, pdf2image) not installed. OCR fallback will be skipped.")

    def ocr_page_image(self, image: Any) -> Tuple[str, bool]:
        """Perform OCR on a PIL Image if available."""
        if not self.is_available or self._tesseract is None:
            return "", True  # ocr_skipped = True

        try:
            text = self._tesseract.image_to_string(image)
            return text, False
        except Exception as e:
            logger.warning("OCR processing error: %s", e)
            return "", True

    def ocr_pdf(self, pdf_path: Path | str) -> Tuple[List[str], bool]:
        """Convert PDF pages to images and run OCR."""
        if not self.is_available or self._pdf2image is None or self._tesseract is None:
            return [], True

        try:
            images = self._pdf2image.convert_from_path(str(pdf_path))
            page_texts = []
            for img in images:
                txt, _ = self.ocr_page_image(img)
                page_texts.append(txt)
            return page_texts, False
        except Exception as e:
            logger.warning("PDF OCR execution failed: %s", e)
            return [], True

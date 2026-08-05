"""OCR extraction that becomes editable TEXT entities in the CAD output."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract

from .models import OCRItem
from .settings import ConversionConfig


@dataclass(slots=True)
class OCRResult:
    items: list[OCRItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        if not self.items:
            return 0.0
        return float(sum(item.confidence for item in self.items) / len(self.items))


def extract_editable_text(image_bgr: np.ndarray, config: ConversionConfig) -> OCRResult:
    """Read text with bounding boxes, preserving it as editable CAD TEXT.

    The Thai + English language request is tried first. If the local Tesseract
    installation does not have Thai trained data, the function falls back to
    English instead of failing the entire CAD conversion.
    """

    if not config.ocr_enabled:
        return OCRResult()

    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale)
    kwargs = {
        "config": "--oem 3 --psm 11",
        "output_type": pytesseract.Output.DICT,
    }

    warnings: list[str] = []
    try:
        data = pytesseract.image_to_data(enhanced, lang=config.ocr_languages, **kwargs)
    except pytesseract.TesseractNotFoundError:
        return OCRResult(
            warnings=[
                "Tesseract OCR is not installed, so no editable text was extracted.",
            ]
        )
    except pytesseract.TesseractError as exc:
        if config.ocr_languages != "eng":
            warnings.append(
                "Requested OCR language data is unavailable; fell back to English. "
                f"Details: {str(exc).splitlines()[0]}"
            )
            try:
                data = pytesseract.image_to_data(enhanced, lang="eng", **kwargs)
            except pytesseract.TesseractError as fallback_error:
                return OCRResult(
                    warnings=warnings
                    + [f"OCR could not run: {str(fallback_error).splitlines()[0]}"]
                )
        else:
            return OCRResult(warnings=[f"OCR could not run: {str(exc).splitlines()[0]}"])

    items: list[OCRItem] = []
    count = len(data["text"])
    for index in range(count):
        text = str(data["text"][index]).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < config.ocr_min_confidence:
            continue

        x = int(data["left"][index])
        y = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])
        if width <= 1 or height <= 1:
            continue
        items.append(
            OCRItem(
                text=text,
                confidence=min(1.0, confidence / 100.0),
                bbox=(x, y, width, height),
            )
        )
    return OCRResult(items=items, warnings=warnings)


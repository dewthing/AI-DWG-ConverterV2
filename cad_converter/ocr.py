"""OCR extraction that becomes editable TEXT entities in the CAD output."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .models import OCRItem
from .settings import ConversionConfig


@dataclass(slots=True)
class OCRResult:
    items: list[OCRItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strategy: str = "none"
    tested_strategies: list[dict[str, object]] = field(default_factory=list)

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

    _configure_tesseract()
    candidates = _ocr_candidates(image_bgr, auto_mode=config.auto_mode)
    candidates = candidates[: max(1, config.ocr_strategy_limit)]
    kwargs = {
        "config": "--oem 3 --psm 11",
        "output_type": pytesseract.Output.DICT,
    }

    warnings: list[str] = []
    language = config.ocr_languages
    results: list[tuple[str, list[OCRItem], float]] = []
    tested: list[dict[str, object]] = []
    for index, (strategy, image) in enumerate(candidates):
        try:
            data = pytesseract.image_to_data(image, lang=language, **kwargs)
        except pytesseract.TesseractNotFoundError:
            return OCRResult(
                warnings=[
                    "Tesseract OCR is not installed, so no editable text was extracted.",
                ]
            )
        except pytesseract.TesseractError as exc:
            if language != "eng" and index == 0:
                warnings.append(
                    "Requested OCR language data is unavailable; fell back to English. "
                    f"Details: {str(exc).splitlines()[0]}"
                )
                language = "eng"
            else:
                warnings.append(
                    f"OCR strategy {strategy} failed: {str(exc).splitlines()[0]}"
                )
                continue
            try:
                data = pytesseract.image_to_data(image, lang=language, **kwargs)
            except pytesseract.TesseractError as fallback_error:
                warnings.append(
                    f"OCR could not run: {str(fallback_error).splitlines()[0]}"
                )
                continue

        items = _items_from_tesseract(data, config)
        score = _candidate_score(items)
        character_count = sum(len(item.text) for item in items)
        tested.append(
            {
                "strategy": strategy,
                "item_count": len(items),
                "character_count": character_count,
                "mean_confidence": round(
                    sum(item.confidence for item in items) / max(len(items), 1),
                    4,
                ),
                "selection_score": round(score, 4),
            }
        )
        results.append((strategy, items, score))

    if not results:
        return OCRResult(warnings=warnings, tested_strategies=tested)
    strategy, items, _ = max(results, key=lambda result: result[2])
    return OCRResult(
        items=items,
        warnings=warnings,
        strategy=strategy,
        tested_strategies=tested,
    )


def _configure_tesseract() -> Path | None:
    """Find Tesseract on PATH or in common Windows installation folders."""

    configured = os.environ.get("AI_CAD_TESSERACT_CMD") or os.environ.get(
        "TESSERACT_CMD"
    )
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    discovered = shutil.which("tesseract")
    if discovered:
        candidates.append(Path(discovered))

    if os.name == "nt":
        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(variable)
            if not base:
                continue
            if variable == "LOCALAPPDATA":
                candidates.append(
                    Path(base) / "Programs" / "Tesseract-OCR" / "tesseract.exe"
                )
            else:
                candidates.append(Path(base) / "Tesseract-OCR" / "tesseract.exe")

    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            pytesseract.pytesseract.tesseract_cmd = str(resolved)
            return resolved
    return None


def _items_from_tesseract(
    data: dict[str, list[object]],
    config: ConversionConfig,
) -> list[OCRItem]:
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
    return items


def _candidate_score(items: list[OCRItem]) -> float:
    if not items:
        return 0.0
    characters = sum(max(1, len(item.text)) for item in items)
    weighted_confidence = sum(
        item.confidence * max(1, len(item.text)) for item in items
    ) / max(characters, 1)
    coverage = min(1.0, characters / 80.0)
    return float(0.82 * weighted_confidence + 0.18 * coverage)


def _ocr_candidates(
    image_bgr: np.ndarray,
    auto_mode: bool,
) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    candidates: list[tuple[str, np.ndarray]] = [("clahe", clahe)]
    if not auto_mode:
        return candidates

    _, otsu = cv2.threshold(
        clahe,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    candidates.append(("clahe_otsu", otsu))

    adaptive = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    candidates.append(("adaptive", adaptive))

    sharpened = cv2.addWeighted(
        gray,
        1.8,
        cv2.GaussianBlur(gray, (0, 0), 1.4),
        -0.8,
        0,
    )
    candidates.append(("unsharp", sharpened))
    return candidates

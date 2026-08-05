"""Extract native vector geometry and text from born-digital PDF pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

from .models import CadEntity, OCRItem


@dataclass(slots=True)
class PDFVectorPage:
    geometry_entities: list[CadEntity] = field(default_factory=list)
    text_items: list[OCRItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_native_pdf_vectors(path: str | Path, dpi: int) -> list[PDFVectorPage]:
    """Extract PDF paths and selectable text in the same pixel grid as rendering.

    This path is used opportunistically: a born-digital PDF retains its original
    vector linework and selectable text, whereas a scanned PDF naturally falls
    back to the OpenCV + OCR pipeline.
    """

    source = Path(path)
    scale = max(dpi, 72) / 72.0
    results: list[PDFVectorPage] = []
    with fitz.open(source) as document:
        for page in document:
            result = PDFVectorPage()
            try:
                for drawing in page.get_drawings():
                    result.geometry_entities.extend(
                        _drawing_entities(drawing, scale)
                    )
                result.text_items.extend(_page_text_items(page, scale))
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                result.warnings.append(
                    f"Native PDF vector extraction was skipped: {str(exc).splitlines()[0]}"
                )
                result.geometry_entities.clear()
                result.text_items.clear()
            results.append(result)
    return results


def _drawing_entities(drawing: dict[str, Any], scale: float) -> list[CadEntity]:
    entities: list[CadEntity] = []
    for item in drawing.get("items", []):
        if not item:
            continue
        command = item[0]
        if command == "l" and len(item) >= 3:
            entities.append(
                CadEntity(
                    kind="LINE",
                    layer="PDF_VECTOR",
                    start=_scaled_point(item[1], scale),
                    end=_scaled_point(item[2], scale),
                    confidence=1.0,
                )
            )
        elif command == "re" and len(item) >= 2:
            rectangle = item[1]
            points = [
                (rectangle.x0 * scale, rectangle.y0 * scale),
                (rectangle.x1 * scale, rectangle.y0 * scale),
                (rectangle.x1 * scale, rectangle.y1 * scale),
                (rectangle.x0 * scale, rectangle.y1 * scale),
            ]
            entities.append(
                CadEntity(
                    kind="LWPOLYLINE",
                    layer="PDF_VECTOR",
                    points=points,
                    closed=True,
                    confidence=1.0,
                )
            )
        elif command == "qu" and len(item) >= 2:
            quad = item[1]
            points = [
                (quad.ul.x * scale, quad.ul.y * scale),
                (quad.ur.x * scale, quad.ur.y * scale),
                (quad.lr.x * scale, quad.lr.y * scale),
                (quad.ll.x * scale, quad.ll.y * scale),
            ]
            entities.append(
                CadEntity(
                    kind="LWPOLYLINE",
                    layer="PDF_VECTOR",
                    points=points,
                    closed=True,
                    confidence=1.0,
                )
            )
        elif command == "c" and len(item) >= 5:
            # Cubic Bezier has no direct primitive in this compact output model;
            # an editable polyline sampled from the curve is safer than losing it.
            entities.append(
                CadEntity(
                    kind="LWPOLYLINE",
                    layer="PDF_VECTOR",
                    points=_sample_cubic_bezier(item[1], item[2], item[3], item[4], scale),
                    closed=False,
                    confidence=0.97,
                )
            )
    return entities


def _page_text_items(page: fitz.Page, scale: float) -> list[OCRItem]:
    items: list[OCRItem] = []
    text = page.get_text("dict")
    for block in text.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                value = str(span.get("text", "")).strip()
                bbox = span.get("bbox")
                if not value or not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = (float(value) * scale for value in bbox)
                items.append(
                    OCRItem(
                        text=value,
                        confidence=1.0,
                        bbox=(
                            int(round(x0)),
                            int(round(y0)),
                            max(1, int(round(x1 - x0))),
                            max(1, int(round(y1 - y0))),
                        ),
                    )
                )
    return items


def _scaled_point(point: Any, scale: float) -> tuple[float, float]:
    return (float(point.x) * scale, float(point.y) * scale)


def _sample_cubic_bezier(
    point0: Any,
    point1: Any,
    point2: Any,
    point3: Any,
    scale: float,
    steps: int = 16,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = index / steps
        inverse = 1.0 - t
        x = (
            inverse**3 * point0.x
            + 3 * inverse**2 * t * point1.x
            + 3 * inverse * t**2 * point2.x
            + t**3 * point3.x
        )
        y = (
            inverse**3 * point0.y
            + 3 * inverse**2 * t * point1.y
            + 3 * inverse * t**2 * point2.y
            + t**3 * point3.y
        )
        result.append((float(x) * scale, float(y) * scale))
    return result


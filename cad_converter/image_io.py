"""Input readers for raster images and multi-page PDFs."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def read_document(path: str | Path, dpi: int = 300) -> list[np.ndarray]:
    """Return one BGR image per source page.

    PDF pages are rendered locally with PyMuPDF, so input drawings are not sent
    to an external service.
    """

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(source, dpi)
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}))
        raise ValueError(f"Unsupported input type {suffix!r}. Supported types: {allowed}")

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {source}")
    return [image]


def _read_pdf(path: Path, dpi: int) -> list[np.ndarray]:
    zoom = max(dpi, 72) / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[np.ndarray] = []

    with fitz.open(path) as document:
        if document.page_count == 0:
            raise ValueError("The PDF does not contain any pages.")
        for page_number in range(document.page_count):
            pixmap = document.load_page(page_number).get_pixmap(
                matrix=matrix,
                alpha=False,
            )
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
                3,
            )
            pages.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return pages


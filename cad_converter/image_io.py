"""Input readers for raster images and multi-page PDFs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image


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


def read_document(
    path: str | Path,
    dpi: int = 300,
    max_page_pixels: int = 25_000_000,
) -> list[np.ndarray]:
    """Return one BGR image per source page.

    PDF pages are rendered locally with PyMuPDF, so input drawings are not sent
    to an external service.
    """

    return list(iter_document(path, dpi=dpi, max_page_pixels=max_page_pixels))


def iter_document(
    path: str | Path,
    dpi: int = 300,
    max_page_pixels: int = 25_000_000,
) -> Iterator[np.ndarray]:
    """Yield pages one at a time to keep multi-page drawing memory bounded."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        yield from _iter_pdf(source, dpi, max_page_pixels)
        return
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}))
        raise ValueError(f"Unsupported input type {suffix!r}. Supported types: {allowed}")

    with Image.open(source) as header:
        _validate_page_size(
            header.width,
            header.height,
            max_page_pixels,
            source.name,
        )
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {source}")
    yield image


def _read_pdf(
    path: Path,
    dpi: int,
    max_page_pixels: int = 25_000_000,
) -> list[np.ndarray]:
    return list(_iter_pdf(path, dpi, max_page_pixels))


def _iter_pdf(
    path: Path,
    dpi: int,
    max_page_pixels: int,
) -> Iterator[np.ndarray]:
    zoom = max(dpi, 72) / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(path) as document:
        if document.page_count == 0:
            raise ValueError("The PDF does not contain any pages.")
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            _validate_page_size(
                int(round(page.rect.width * zoom)),
                int(round(page.rect.height * zoom)),
                max_page_pixels,
                f"{path.name} page {page_number + 1}",
            )
            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
                3,
            )
            yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _validate_page_size(
    width: int,
    height: int,
    max_page_pixels: int,
    label: str,
) -> None:
    if max_page_pixels <= 0:
        return
    pixels = width * height
    if pixels <= max_page_pixels:
        return
    actual = pixels / 1_000_000.0
    allowed = max_page_pixels / 1_000_000.0
    raise ValueError(
        f"{label} would use {actual:.1f} megapixels, above the {allowed:.1f} MP "
        "safety limit. Lower PDF DPI or increase max_page_pixels for a machine "
        "with enough memory."
    )

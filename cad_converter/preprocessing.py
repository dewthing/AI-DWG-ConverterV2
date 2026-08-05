"""Generate complementary image-processing candidates for engineering drawings."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessVariant:
    name: str
    mask: np.ndarray  # uint8, 0 for background and 255 for ink


def reference_ink_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Build a stable reference mask used only for quality comparison."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return _remove_border_noise(mask)


def generate_variants(image_bgr: np.ndarray, iteration: int = 0) -> list[PreprocessVariant]:
    """Produce different masks for dark, faint, coloured, and scanned drawings.

    Each later iteration introduces more aggressive enhancement techniques. The
    converter evaluates the results rather than assuming one preprocessing
    method is correct for every drawing.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    denoised = cv2.fastNlMeansDenoising(gray, None, 9, 7, 21)
    variants: list[PreprocessVariant] = []

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(PreprocessVariant("otsu", _remove_border_noise(otsu)))

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    variants.append(PreprocessVariant("adaptive", _remove_border_noise(adaptive)))

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    _, clahe_otsu = cv2.threshold(
        clahe,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    variants.append(PreprocessVariant("clahe_otsu", _remove_border_noise(clahe_otsu)))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    dark_ink = cv2.inRange(hsv, (0, 0, 0), (180, 255, 190))
    coloured_ink = cv2.inRange(hsv, (0, 45, 0), (180, 255, 255))
    colour_ink = cv2.bitwise_or(dark_ink, coloured_ink)
    variants.append(PreprocessVariant("colour_ink", _remove_border_noise(colour_ink)))

    edges = cv2.Canny(blurred, 40, 140, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    variants.append(PreprocessVariant("edge_preserving", _remove_border_noise(edges)))

    if iteration >= 1:
        gamma_light = _gamma(gray, 1.55)
        _, light_otsu = cv2.threshold(
            gamma_light,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        variants.append(PreprocessVariant("gamma_light_otsu", _remove_border_noise(light_otsu)))

        gamma_dark = _gamma(gray, 0.62)
        _, dark_otsu = cv2.threshold(
            gamma_dark,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        variants.append(PreprocessVariant("gamma_dark_otsu", _remove_border_noise(dark_otsu)))

        closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        variants.append(PreprocessVariant("adaptive_closed", _remove_border_noise(closed)))

    if iteration >= 2:
        sharpened = cv2.addWeighted(
            gray,
            1.7,
            cv2.GaussianBlur(gray, (0, 0), 2.0),
            -0.7,
            0,
        )
        _, sharp_otsu = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        variants.append(PreprocessVariant("unsharp_otsu", _remove_border_noise(sharp_otsu)))

        hybrid = cv2.bitwise_or(otsu, edges)
        variants.append(PreprocessVariant("otsu_edges_hybrid", _remove_border_noise(hybrid)))

    return _unique_variants(variants)


def _gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    lookup = np.array(
        [((index / 255.0) ** gamma) * 255.0 for index in np.arange(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(gray, lookup)


def _remove_border_noise(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.copy()
    height, width = cleaned.shape[:2]
    border = max(1, min(height, width) // 500)
    cleaned[:border, :] = 0
    cleaned[-border:, :] = 0
    cleaned[:, :border] = 0
    cleaned[:, -border:] = 0
    return cleaned


def _unique_variants(variants: list[PreprocessVariant]) -> list[PreprocessVariant]:
    unique: list[PreprocessVariant] = []
    seen: set[bytes] = set()
    for variant in variants:
        fingerprint = variant.mask[::16, ::16].tobytes()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(variant)
    return unique


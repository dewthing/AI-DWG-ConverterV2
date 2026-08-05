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
    """Build a stable reference mask used only for quality comparison.

    Otsu alone can discard pale red, green, blue, and orange CAD layers.  The
    saturation mask keeps that born-digital coloured linework in the QA target
    without changing the preprocessing candidates themselves.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, grayscale_ink = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    dark_ink = cv2.inRange(hsv, (0, 0, 0), (180, 255, 210))
    coloured_ink = cv2.inRange(hsv, (0, 30, 0), (180, 255, 255))
    mask = cv2.bitwise_or(
        grayscale_ink,
        cv2.bitwise_or(dark_ink, coloured_ink),
    )
    return _remove_border_noise(mask)


def generate_variants(
    image_bgr: np.ndarray,
    iteration: int = 0,
    selected_names: list[str] | None = None,
) -> list[PreprocessVariant]:
    """Produce different masks for dark, faint, coloured, and scanned drawings.

    Each later iteration introduces more aggressive enhancement techniques. The
    converter evaluates the results rather than assuming one preprocessing
    method is correct for every drawing.
    """

    available = {
        "otsu",
        "adaptive",
        "clahe_otsu",
        "colour_ink",
        "edge_preserving",
    }
    if iteration >= 1:
        available.update(
            {
                "gamma_light_otsu",
                "gamma_dark_otsu",
                "adaptive_closed",
                "background_normalized",
                "bilateral_adaptive",
            }
        )
    if iteration >= 2:
        available.update({"unsharp_otsu", "otsu_edges_hybrid", "blackhat_ink"})
    wanted = available if selected_names is None else available.intersection(selected_names)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants: list[PreprocessVariant] = []

    otsu: np.ndarray | None = None
    edges: np.ndarray | None = None
    adaptive: np.ndarray | None = None

    if {"otsu", "otsu_edges_hybrid"}.intersection(wanted):
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        if "otsu" in wanted:
            variants.append(PreprocessVariant("otsu", _remove_border_noise(otsu)))

    if {"adaptive", "adaptive_closed"}.intersection(wanted):
        denoised = cv2.fastNlMeansDenoising(gray, None, 9, 7, 21)
        adaptive = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )
        if "adaptive" in wanted:
            variants.append(PreprocessVariant("adaptive", _remove_border_noise(adaptive)))

    if "clahe_otsu" in wanted:
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
        _, clahe_otsu = cv2.threshold(
            clahe,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        variants.append(PreprocessVariant("clahe_otsu", _remove_border_noise(clahe_otsu)))

    if "colour_ink" in wanted:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        dark_ink = cv2.inRange(hsv, (0, 0, 0), (180, 255, 190))
        coloured_ink = cv2.inRange(hsv, (0, 45, 0), (180, 255, 255))
        colour_ink = cv2.bitwise_or(dark_ink, coloured_ink)
        variants.append(PreprocessVariant("colour_ink", _remove_border_noise(colour_ink)))

    if {"edge_preserving", "otsu_edges_hybrid"}.intersection(wanted):
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 140, apertureSize=3)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        if "edge_preserving" in wanted:
            variants.append(PreprocessVariant("edge_preserving", _remove_border_noise(edges)))

    if iteration >= 1:
        if "gamma_light_otsu" in wanted:
            gamma_light = _gamma(gray, 1.55)
            _, light_otsu = cv2.threshold(
                gamma_light,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            variants.append(
                PreprocessVariant("gamma_light_otsu", _remove_border_noise(light_otsu))
            )

        if "gamma_dark_otsu" in wanted:
            gamma_dark = _gamma(gray, 0.62)
            _, dark_otsu = cv2.threshold(
                gamma_dark,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            variants.append(
                PreprocessVariant("gamma_dark_otsu", _remove_border_noise(dark_otsu))
            )

        if "adaptive_closed" in wanted:
            if adaptive is None:
                denoised = cv2.fastNlMeansDenoising(gray, None, 9, 7, 21)
                adaptive = cv2.adaptiveThreshold(
                    denoised,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    31,
                    7,
                )
            closed = cv2.morphologyEx(
                adaptive,
                cv2.MORPH_CLOSE,
                np.ones((2, 2), np.uint8),
            )
            variants.append(PreprocessVariant("adaptive_closed", _remove_border_noise(closed)))

        if "background_normalized" in wanted:
            background_kernel = np.ones((31, 31), dtype=np.uint8)
            background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, background_kernel)
            normalized_background = cv2.divide(gray, np.maximum(background, 1), scale=255)
            _, normalized_otsu = cv2.threshold(
                normalized_background,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            variants.append(
                PreprocessVariant(
                    "background_normalized",
                    _remove_border_noise(normalized_otsu),
                )
            )

        if "bilateral_adaptive" in wanted:
            bilateral = cv2.bilateralFilter(gray, 7, 45, 45)
            bilateral_adaptive = cv2.adaptiveThreshold(
                bilateral,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                7,
            )
            variants.append(
                PreprocessVariant(
                    "bilateral_adaptive",
                    _remove_border_noise(bilateral_adaptive),
                )
            )

    if iteration >= 2:
        if "unsharp_otsu" in wanted:
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

        if "otsu_edges_hybrid" in wanted:
            if otsu is None or edges is None:
                raise RuntimeError("Hybrid preprocessing dependencies were not generated.")
            hybrid = cv2.bitwise_or(otsu, edges)
            variants.append(PreprocessVariant("otsu_edges_hybrid", _remove_border_noise(hybrid)))

        if "blackhat_ink" in wanted:
            blackhat = cv2.morphologyEx(
                gray,
                cv2.MORPH_BLACKHAT,
                cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
            )
            _, blackhat_ink = cv2.threshold(
                blackhat,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            variants.append(
                PreprocessVariant("blackhat_ink", _remove_border_noise(blackhat_ink))
            )

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

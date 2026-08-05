"""Configuration objects and constants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConversionConfig:
    """Tunable conversion settings.

    pixels_per_unit is intentionally 1.0 by default, matching the user's
    preferred 1 source pixel = 1 CAD drawing unit workflow. Scale calibration
    can be introduced later without changing the detected geometry.
    """

    pixels_per_unit: float = 1.0
    auto_pdf_scale: bool = True
    pdf_dpi: int = 300
    max_page_pixels: int = 25_000_000
    auto_mode: bool = True
    max_iterations: int = 3
    desired_score: float = 0.92
    min_score_improvement: float = 0.003
    plateau_patience: int = 2
    min_line_length: int = 18
    max_line_gap: int = 12
    hough_threshold: int = 34
    min_circle_radius: int = 6
    max_circle_radius: int = 0  # 0 means derive from input dimensions.
    contour_min_area: float = 100.0
    contour_epsilon_ratio: float = 0.012
    max_entities: int = 2400
    ocr_enabled: bool = True
    ocr_languages: str = "tha+eng"
    ocr_min_confidence: float = 30.0
    ocr_strategy_limit: int = 3
    text_height_multiplier: float = 0.9
    cad_text_style: str = "OCR_TEXT"
    cad_text_font: str = "Arial.ttf"
    export_dwg: bool = False
    oda_converter_path: str | None = None
    dwg_version: str = "ACAD2018"

"""Raster-to-geometry detection and raster reconstruction for quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot

import cv2
import numpy as np

from .models import CadEntity, OCRItem
from .settings import ConversionConfig


@dataclass(slots=True)
class GeometryQuality:
    geometric_score: float
    tolerant_f1: float
    exact_iou: float


class RasterVectorizer:
    """Detect editable CAD primitives from an ink mask.

    This is intentionally a hybrid vectorizer: Hough transforms preserve common
    CAD primitives, while contour approximation keeps symbols and irregular
    closed outlines as editable LWPOLYLINE entities.
    """

    def __init__(self, config: ConversionConfig) -> None:
        self.config = config

    def vectorize(self, ink_mask: np.ndarray) -> list[CadEntity]:
        if ink_mask.ndim != 2:
            raise ValueError("RasterVectorizer expects a one-channel ink mask.")

        entities: list[CadEntity] = []
        circles = self._detect_circles(ink_mask)
        entities.extend(circles)
        entities.extend(self._detect_lines(ink_mask))

        remaining = max(self.config.max_entities - len(entities), 0)
        if remaining:
            entities.extend(self._detect_polylines(ink_mask, circles, remaining))

        return entities[: self.config.max_entities]

    def _detect_lines(self, mask: np.ndarray) -> list[CadEntity]:
        raw_lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 360.0,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )
        if raw_lines is None:
            return []

        candidates: list[tuple[float, float, float, float, float, float]] = []
        # OpenCV 4 commonly returns shape (N, 1, 4), while OpenCV 5 can
        # return (N, 4). Flattening makes the converter portable across both.
        for line in np.asarray(raw_lines).reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in line)
            length = hypot(x2 - x1, y2 - y1)
            if length < self.config.min_line_length:
                continue
            angle = degrees(atan2(y2 - y1, x2 - x1)) % 180.0
            candidates.append((x1, y1, x2, y2, length, angle))

        candidates.sort(key=lambda item: item[4], reverse=True)
        accepted: list[tuple[float, float, float, float, float, float]] = []
        for candidate in candidates:
            if not any(self._is_same_line(candidate, prior) for prior in accepted):
                accepted.append(candidate)
            if len(accepted) >= self.config.max_entities:
                break

        return [
            CadEntity(
                kind="LINE",
                layer="GEOMETRY",
                start=(line[0], line[1]),
                end=(line[2], line[3]),
                confidence=min(1.0, 0.45 + line[4] / 250.0),
            )
            for line in accepted
        ]

    @staticmethod
    def _is_same_line(
        first: tuple[float, float, float, float, float, float],
        second: tuple[float, float, float, float, float, float],
    ) -> bool:
        angle_delta = min(abs(first[5] - second[5]), 180.0 - abs(first[5] - second[5]))
        if angle_delta > 3.0 or abs(first[4] - second[4]) > 18.0:
            return False

        first_midpoint = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        second_midpoint = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
        return hypot(
            first_midpoint[0] - second_midpoint[0],
            first_midpoint[1] - second_midpoint[1],
        ) < 10.0

    def _detect_circles(self, mask: np.ndarray) -> list[CadEntity]:
        height, width = mask.shape[:2]
        max_radius = self.config.max_circle_radius or max(8, min(height, width) // 4)
        softened = cv2.GaussianBlur(mask, (5, 5), 1.2)
        raw_circles = cv2.HoughCircles(
            softened,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(14, self.config.min_circle_radius * 2),
            param1=120,
            param2=18,
            minRadius=self.config.min_circle_radius,
            maxRadius=max_radius,
        )
        if raw_circles is None:
            return []

        support_mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8))
        accepted: list[CadEntity] = []
        for x, y, radius in np.round(raw_circles[0, :]).astype(int):
            inside_ratio, support_ratio = self._circle_support(
                support_mask,
                x,
                y,
                radius,
            )
            # HoughCircles often proposes circles around rectangle corners and
            # character fragments.  A real circle needs ink around most of its
            # circumference and should be substantially inside the page.
            if inside_ratio < 0.90 or support_ratio < 0.52:
                continue
            entity = CadEntity(
                kind="CIRCLE",
                layer="GEOMETRY",
                center=(float(x), float(y)),
                radius=float(radius),
                confidence=min(0.99, 0.45 + 0.54 * support_ratio),
                extra={"circumference_support": round(support_ratio, 4)},
            )
            if not any(self._is_same_circle(entity, existing) for existing in accepted):
                accepted.append(entity)
        return accepted

    @staticmethod
    def _circle_support(
        mask: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int,
        samples: int = 180,
    ) -> tuple[float, float]:
        height, width = mask.shape[:2]
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        xs = np.rint(center_x + radius * np.cos(angles)).astype(np.int32)
        ys = np.rint(center_y + radius * np.sin(angles)).astype(np.int32)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        inside_count = int(np.count_nonzero(inside))
        if inside_count == 0:
            return 0.0, 0.0
        support = int(np.count_nonzero(mask[ys[inside], xs[inside]]))
        return inside_count / samples, support / inside_count

    @staticmethod
    def _is_same_circle(first: CadEntity, second: CadEntity) -> bool:
        if first.center is None or second.center is None:
            return False
        return (
            hypot(first.center[0] - second.center[0], first.center[1] - second.center[1]) < 8
            and abs((first.radius or 0) - (second.radius or 0)) < 6
        )

    def _detect_polylines(
        self,
        mask: np.ndarray,
        circles: list[CadEntity],
        maximum: int,
    ) -> list[CadEntity]:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        height, width = mask.shape[:2]
        results: list[CadEntity] = []

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if len(results) >= maximum:
                break
            area = float(cv2.contourArea(contour))
            if area < self.config.contour_min_area:
                continue

            x, y, bounding_width, bounding_height = cv2.boundingRect(contour)
            if bounding_width >= width * 0.98 and bounding_height >= height * 0.98:
                continue
            if min(bounding_width, bounding_height) < 7:
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = max(1.0, perimeter * self.config.contour_epsilon_ratio)
            approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(approximation) < 3:
                continue

            candidate_center = ((x + bounding_width / 2.0), (y + bounding_height / 2.0))
            if self._matches_known_circle(candidate_center, bounding_width, bounding_height, circles):
                continue

            points = [(float(point[0]), float(point[1])) for point in approximation]
            results.append(
                CadEntity(
                    kind="LWPOLYLINE",
                    layer="SYMBOLS",
                    points=points,
                    closed=True,
                    confidence=min(0.95, 0.45 + area / max(width * height, 1) * 25.0),
                )
            )
        return results

    @staticmethod
    def _matches_known_circle(
        center: tuple[float, float],
        width: float,
        height: float,
        circles: list[CadEntity],
    ) -> bool:
        possible_radius = (width + height) / 4.0
        for circle in circles:
            if circle.center is None or circle.radius is None:
                continue
            if (
                hypot(center[0] - circle.center[0], center[1] - circle.center[1]) < 8.0
                and abs(possible_radius - circle.radius) < 8.0
            ):
                return True
        return False


def render_entities(
    entities: list[CadEntity],
    shape: tuple[int, int],
    line_thickness: int = 1,
) -> np.ndarray:
    """Render geometry entities to an ink mask for QA scoring."""

    height, width = shape
    rendered = np.zeros((height, width), dtype=np.uint8)

    for entity in entities:
        if entity.kind == "LINE" and entity.start and entity.end:
            cv2.line(
                rendered,
                _as_int_point(entity.start),
                _as_int_point(entity.end),
                255,
                line_thickness,
                cv2.LINE_AA,
            )
        elif entity.kind == "CIRCLE" and entity.center and entity.radius:
            cv2.circle(
                rendered,
                _as_int_point(entity.center),
                max(1, int(round(entity.radius))),
                255,
                line_thickness,
                cv2.LINE_AA,
            )
        elif entity.kind == "ARC" and entity.center and entity.radius is not None:
            cv2.ellipse(
                rendered,
                _as_int_point(entity.center),
                (max(1, int(entity.radius)), max(1, int(entity.radius))),
                0,
                entity.start_angle or 0,
                entity.end_angle or 360,
                255,
                line_thickness,
                cv2.LINE_AA,
            )
        elif entity.kind == "LWPOLYLINE" and entity.points:
            points = np.array([_as_int_point(point) for point in entity.points], dtype=np.int32)
            cv2.polylines(
                rendered,
                [points],
                entity.closed,
                255,
                line_thickness,
                cv2.LINE_AA,
            )

    return rendered


def compare_geometry(reference_mask: np.ndarray, rendered_mask: np.ndarray) -> GeometryQuality:
    """Score generated primitives against the raster source with small tolerance."""

    reference = reference_mask > 0
    rendered = rendered_mask > 0
    kernel = np.ones((5, 5), dtype=np.uint8)
    dilated_reference = cv2.dilate(reference.astype(np.uint8), kernel) > 0
    dilated_rendered = cv2.dilate(rendered.astype(np.uint8), kernel) > 0

    true_positive_precision = int(np.count_nonzero(rendered & dilated_reference))
    true_positive_recall = int(np.count_nonzero(reference & dilated_rendered))
    predicted_count = int(np.count_nonzero(rendered))
    reference_count = int(np.count_nonzero(reference))

    precision = true_positive_precision / max(predicted_count, 1)
    recall = true_positive_recall / max(reference_count, 1)
    tolerant_f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)

    intersection = int(np.count_nonzero(reference & rendered))
    union = int(np.count_nonzero(reference | rendered))
    exact_iou = intersection / max(union, 1)
    score = 0.82 * tolerant_f1 + 0.18 * exact_iou
    return GeometryQuality(
        geometric_score=float(score),
        tolerant_f1=float(tolerant_f1),
        exact_iou=float(exact_iou),
    )


def make_overlay(
    source_bgr: np.ndarray,
    reference_mask: np.ndarray,
    rendered_mask: np.ndarray,
    text_items: list[OCRItem] | None = None,
) -> np.ndarray:
    """Return a readable side-by-side source / reconstructed / difference preview."""

    height, width = reference_mask.shape
    reconstruction = cv2.cvtColor(255 - rendered_mask, cv2.COLOR_GRAY2BGR)
    overlay = np.full((height, width, 3), 255, dtype=np.uint8)

    reference = reference_mask > 0
    rendered = rendered_mask > 0
    overlap = reference & rendered
    only_source = reference & ~rendered
    only_vector = rendered & ~reference
    overlay[overlap] = (0, 130, 0)
    overlay[only_source] = (0, 0, 220)
    overlay[only_vector] = (220, 120, 0)
    _draw_ocr_preview(reconstruction, text_items or [], include_text=True)
    _draw_ocr_preview(overlay, text_items or [], include_text=False)

    source = source_bgr.copy()
    if source.shape[:2] != (height, width):
        source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    return np.hstack(
        (
            _with_panel_header(source, "SOURCE"),
            _with_panel_header(reconstruction, "CAD RECONSTRUCTION"),
            _with_panel_header(
                overlay,
                "QA: GREEN MATCH / RED MISSING / BLUE EXTRA / MAGENTA OCR",
            ),
        )
    )


def _draw_ocr_preview(
    image: np.ndarray,
    items: list[OCRItem],
    include_text: bool,
) -> None:
    height, width = image.shape[:2]
    for item in items:
        x, y, item_width, item_height = item.bbox
        left = max(0, min(width - 1, x))
        top = max(0, min(height - 1, y))
        right = max(left, min(width - 1, x + item_width))
        bottom = max(top, min(height - 1, y + item_height))
        cv2.rectangle(image, (left, top), (right, bottom), (180, 0, 180), 1)
        if not include_text:
            continue
        ascii_text = item.text.encode("ascii", errors="ignore").decode("ascii").strip()
        label = ascii_text[:28] if ascii_text else "OCR TEXT"
        cv2.putText(
            image,
            label,
            (left, max(12, top - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 0, 180),
            1,
            cv2.LINE_AA,
        )


def _with_panel_header(image: np.ndarray, title: str) -> np.ndarray:
    header_height = 42
    header = np.full((header_height, image.shape[1], 3), 245, dtype=np.uint8)
    font_scale = max(0.42, min(0.72, image.shape[1] / 900.0))
    cv2.putText(
        header,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def _as_int_point(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))

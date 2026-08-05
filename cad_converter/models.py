"""Shared data structures for the conversion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


EntityKind = Literal[
    "LINE",
    "CIRCLE",
    "ARC",
    "LWPOLYLINE",
    "HATCH",
    "TEXT",
]


@dataclass(slots=True)
class CadEntity:
    """A CAD entity stored in source-image pixel coordinates.

    Coordinates deliberately remain in image coordinates until export. This makes
    raster QA simple and lets the exporter apply a single, consistent Y-axis flip.
    """

    kind: EntityKind
    layer: str
    confidence: float = 1.0
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    center: tuple[float, float] | None = None
    radius: float | None = None
    start_angle: float | None = None
    end_angle: float | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    boundary_paths: list[list[tuple[float, float]]] = field(default_factory=list)
    closed: bool = False
    text: str | None = None
    height: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OCRItem:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    rotation: float = 0.0
    font: str | None = None
    color: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateMetrics:
    geometric_score: float
    tolerant_f1: float
    exact_iou: float
    ocr_score: float
    learned_score: float
    final_score: float
    ink_ratio: float
    line_count: int
    circle_count: int
    polyline_count: int
    text_count: int
    fragmentation: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class CandidateResult:
    name: str
    iteration: int
    entities: list[CadEntity]
    text_items: list[OCRItem]
    metrics: CandidateMetrics
    preview_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iteration": self.iteration,
            "entities": [entity.to_dict() for entity in self.entities],
            "text_items": [item.to_dict() for item in self.text_items],
            "metrics": self.metrics.to_dict(),
            "preview_path": str(self.preview_path) if self.preview_path else None,
        }

"""End-to-end iterative image/PDF to editable CAD conversion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .exporter import export_dwg_with_oda, export_dxf
from .image_io import read_document
from .learning import FeedbackLearner, TrainingSummary
from .models import CadEntity, CandidateMetrics, CandidateResult, OCRItem
from .ocr import OCRResult, extract_editable_text
from .pdf_vector import PDFVectorPage, extract_native_pdf_vectors
from .preprocessing import generate_variants, reference_ink_mask
from .settings import ConversionConfig
from .vectorizer import RasterVectorizer, compare_geometry, make_overlay, render_entities


@dataclass(slots=True)
class PageResult:
    source_name: str
    page_number: int
    candidate: CandidateResult
    dxf_path: Path
    dwg_path: Path | None
    preview_path: Path
    reference_path: Path
    report_path: Path
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "page_number": self.page_number,
            "candidate": self.candidate.to_dict(),
            "dxf_path": str(self.dxf_path),
            "dwg_path": str(self.dwg_path) if self.dwg_path else None,
            "preview_path": str(self.preview_path),
            "reference_path": str(self.reference_path),
            "report_path": str(self.report_path),
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class DocumentResult:
    source_path: Path
    output_directory: Path
    pages: list[PageResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "output_directory": str(self.output_directory),
            "pages": [page.to_dict() for page in self.pages],
        }


class CADConverter:
    """Convert each input page using iterative preprocessing and QA scoring."""

    def __init__(
        self,
        config: ConversionConfig | None = None,
        feedback_path: str | Path = "data/feedback.jsonl",
    ) -> None:
        self.config = config or ConversionConfig()
        self.vectorizer = RasterVectorizer(self.config)
        self.learner = FeedbackLearner(feedback_path)

    def convert(
        self,
        input_path: str | Path,
        output_directory: str | Path,
    ) -> DocumentResult:
        source = Path(input_path).expanduser().resolve()
        target_dir = Path(output_directory).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        pages = read_document(source, dpi=self.config.pdf_dpi)
        native_pdf_pages: list[PDFVectorPage] = []
        if source.suffix.lower() == ".pdf":
            native_pdf_pages = extract_native_pdf_vectors(source, self.config.pdf_dpi)
        safe_stem = _safe_stem(source.stem)

        results: list[PageResult] = []
        for page_number, image_bgr in enumerate(pages, start=1):
            prefix = f"{safe_stem}_page_{page_number:03d}"
            results.append(
                self._convert_page(
                    image_bgr=image_bgr,
                    source_name=source.name,
                    page_number=page_number,
                    prefix=prefix,
                    output_directory=target_dir,
                    native_pdf_page=(
                        native_pdf_pages[page_number - 1]
                        if page_number <= len(native_pdf_pages)
                        else None
                    ),
                )
            )

        document_result = DocumentResult(
            source_path=source,
            output_directory=target_dir,
            pages=results,
        )
        manifest_path = target_dir / "conversion_manifest.json"
        manifest_path.write_text(
            json.dumps(document_result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return document_result

    def save_feedback(
        self,
        page: PageResult,
        score_percent: float,
        accepted: bool,
        note: str = "",
    ) -> TrainingSummary:
        return self.learner.record_feedback(
            source_name=page.source_name,
            candidate_name=page.candidate.name,
            metrics=page.candidate.metrics,
            score_percent=score_percent,
            accepted=accepted,
            note=note,
        )

    def _convert_page(
        self,
        image_bgr: np.ndarray,
        source_name: str,
        page_number: int,
        prefix: str,
        output_directory: Path,
        native_pdf_page: PDFVectorPage | None = None,
    ) -> PageResult:
        height, width = image_bgr.shape[:2]
        warnings: list[str] = []
        reference = reference_ink_mask(image_bgr)
        native_geometry: list[CadEntity] = []
        if native_pdf_page is not None:
            native_geometry = native_pdf_page.geometry_entities
            warnings.extend(native_pdf_page.warnings)
        if native_pdf_page is not None and native_pdf_page.text_items:
            ocr_result = OCRResult(items=native_pdf_page.text_items)
        else:
            ocr_result = extract_editable_text(image_bgr, self.config)
        warnings.extend(ocr_result.warnings)
        text_entities = _text_entities(ocr_result.items, self.config)
        geometry_reference = _remove_text_from_mask(reference, ocr_result.items)
        reference_path = output_directory / f"{prefix}_geometry_reference.png"
        cv2.imwrite(str(reference_path), geometry_reference)

        best: CandidateResult | None = None
        seen_masks: set[str] = set()
        candidates: list[CandidateResult] = []
        if native_geometry:
            native_candidate = self._score_entity_candidate(
                variant_name="native_pdf_vectors",
                iteration=0,
                geometry_entities=native_geometry,
                reference=geometry_reference,
                text_entities=text_entities,
                ocr_result=ocr_result,
                ink_ratio=float(np.count_nonzero(geometry_reference) / geometry_reference.size),
            )
            candidates.append(native_candidate)
            best = native_candidate

        for iteration in range(max(1, self.config.max_iterations)):
            for variant in generate_variants(image_bgr, iteration):
                fingerprint = _mask_fingerprint(variant.mask)
                if fingerprint in seen_masks:
                    continue
                seen_masks.add(fingerprint)
                candidate = self._evaluate_candidate(
                    variant_name=variant.name,
                    iteration=iteration,
                    mask=_remove_text_from_mask(variant.mask, ocr_result.items),
                    reference=geometry_reference,
                    text_entities=text_entities,
                    ocr_result=ocr_result,
                )
                candidates.append(candidate)
                if best is None or candidate.metrics.final_score > best.metrics.final_score:
                    best = candidate

            if best is not None and best.metrics.final_score >= self.config.desired_score:
                break

        if best is None:
            raise RuntimeError("No preprocessing candidates could be generated.")

        rendered = render_entities(
            [entity for entity in best.entities if entity.kind != "TEXT"],
            (height, width),
        )
        overlay = make_overlay(image_bgr, geometry_reference, rendered)
        preview_path = output_directory / f"{prefix}_qa_preview.png"
        cv2.imwrite(str(preview_path), overlay)
        best.preview_path = preview_path

        dxf_path = export_dxf(
            best.entities,
            (height, width),
            output_directory / f"{prefix}.dxf",
            self.config,
        )
        dwg_path: Path | None = None
        if self.config.export_dwg:
            dwg_result = export_dwg_with_oda(
                dxf_path,
                output_directory / f"{prefix}.dwg",
                self.config,
            )
            dwg_path = dwg_result.path
            if dwg_result.warning:
                warnings.append(dwg_result.warning)

        report_path = output_directory / f"{prefix}_report.json"
        report_payload = {
            "source_name": source_name,
            "page_number": page_number,
            "image_size": {"width": width, "height": height},
            "qa_reference": "Text regions are excluded from geometry QA and assessed separately by OCR.",
            "settings": asdict(self.config),
            "best_candidate": best.to_dict(),
            "tested_candidates": [
                {
                    "name": candidate.name,
                    "iteration": candidate.iteration,
                    "metrics": candidate.metrics.to_dict(),
                }
                for candidate in sorted(
                    candidates,
                    key=lambda item: item.metrics.final_score,
                    reverse=True,
                )
            ],
            "warnings": warnings,
        }
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return PageResult(
            source_name=source_name,
            page_number=page_number,
            candidate=best,
            dxf_path=dxf_path,
            dwg_path=dwg_path,
            preview_path=preview_path,
            reference_path=reference_path,
            report_path=report_path,
            warnings=warnings,
        )

    def _evaluate_candidate(
        self,
        variant_name: str,
        iteration: int,
        mask: np.ndarray,
        reference: np.ndarray,
        text_entities: list[CadEntity],
        ocr_result: OCRResult,
    ) -> CandidateResult:
        geometry_entities = self.vectorizer.vectorize(mask)
        return self._score_entity_candidate(
            variant_name=variant_name,
            iteration=iteration,
            geometry_entities=geometry_entities,
            reference=reference,
            text_entities=text_entities,
            ocr_result=ocr_result,
            ink_ratio=float(np.count_nonzero(mask) / mask.size),
        )

    def _score_entity_candidate(
        self,
        variant_name: str,
        iteration: int,
        geometry_entities: list[CadEntity],
        reference: np.ndarray,
        text_entities: list[CadEntity],
        ocr_result: OCRResult,
        ink_ratio: float,
    ) -> CandidateResult:
        rendered = render_entities(geometry_entities, reference.shape)
        geometry_quality = compare_geometry(reference, rendered)

        line_count = sum(entity.kind == "LINE" for entity in geometry_entities)
        circle_count = sum(entity.kind == "CIRCLE" for entity in geometry_entities)
        polyline_count = sum(entity.kind == "LWPOLYLINE" for entity in geometry_entities)
        ink_pixels = max(int(ink_ratio * reference.size), 1)
        fragmentation = len(geometry_entities) / max(float(np.sqrt(ink_pixels)), 1.0)
        # No recognised text does not count as an OCR failure. It simply leaves
        # quality selection to the geometric comparison.
        ocr_score = ocr_result.quality_score if ocr_result.items else 1.0

        provisional = CandidateMetrics(
            geometric_score=geometry_quality.geometric_score,
            tolerant_f1=geometry_quality.tolerant_f1,
            exact_iou=geometry_quality.exact_iou,
            ocr_score=ocr_score,
            learned_score=geometry_quality.geometric_score,
            final_score=0.0,
            ink_ratio=ink_ratio,
            line_count=line_count,
            circle_count=circle_count,
            polyline_count=polyline_count,
            text_count=len(text_entities),
            fragmentation=fragmentation,
        )
        prediction = self.learner.predict(provisional)
        learned_score = prediction if prediction is not None else geometry_quality.geometric_score
        if prediction is None:
            final_score = 0.90 * geometry_quality.geometric_score + 0.10 * ocr_score
        else:
            final_score = (
                0.76 * geometry_quality.geometric_score
                + 0.10 * ocr_score
                + 0.14 * prediction
            )

        metrics = CandidateMetrics(
            geometric_score=geometry_quality.geometric_score,
            tolerant_f1=geometry_quality.tolerant_f1,
            exact_iou=geometry_quality.exact_iou,
            ocr_score=ocr_score,
            learned_score=learned_score,
            final_score=float(final_score),
            ink_ratio=provisional.ink_ratio,
            line_count=line_count,
            circle_count=circle_count,
            polyline_count=polyline_count,
            text_count=len(text_entities),
            fragmentation=float(fragmentation),
        )
        return CandidateResult(
            name=f"{variant_name} / iteration {iteration + 1}",
            iteration=iteration + 1,
            entities=[*geometry_entities, *text_entities],
            text_items=ocr_result.items,
            metrics=metrics,
        )


def _text_entities(items: list[OCRItem], config: ConversionConfig) -> list[CadEntity]:
    entities: list[CadEntity] = []
    for item in items:
        _, _, _, height = item.bbox
        entities.append(
            CadEntity(
                kind="TEXT",
                layer="TEXT",
                confidence=item.confidence,
                text=item.text,
                height=max(1.0, height * config.text_height_multiplier),
                bbox=item.bbox,
            )
        )
    return entities


def _mask_fingerprint(mask: np.ndarray) -> str:
    small = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes()).hexdigest()


def _remove_text_from_mask(mask: np.ndarray, text_items: list[OCRItem]) -> np.ndarray:
    """Prevent text outlines from being mistaken for geometry during vector QA."""

    if not text_items:
        return mask
    result = mask.copy()
    height, width = result.shape[:2]
    for item in text_items:
        x, y, item_width, item_height = item.bbox
        padding = max(1, min(item_height // 8, 4))
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(width - 1, x + item_width + padding)
        bottom = min(height - 1, y + item_height + padding)
        cv2.rectangle(result, (left, top), (right, bottom), 0, thickness=-1)
    return result


def _safe_stem(stem: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    return cleaned.strip("_") or "drawing"

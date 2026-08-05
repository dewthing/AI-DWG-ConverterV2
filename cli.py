"""Command-line entry point for AI CAD Converter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cad_converter import CADConverter, ConversionConfig
from cad_converter.archive import create_output_archive
from cad_converter.learning import FeedbackLearner
from cad_converter.models import CandidateMetrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert image/PDF engineering drawings into editable DXF/DWG.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert an image or PDF.")
    convert.add_argument("input", type=Path, help="Source image or PDF")
    convert.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory for CAD files and QA previews",
    )
    convert.add_argument("--dwg", action="store_true", help="Also request DWG output via ODA")
    convert.add_argument(
        "--manual",
        action="store_true",
        help="Disable automatic image profiling and evaluate all available strategies",
    )
    convert.add_argument("--oda-path", help="Path to ODAFileConverter executable")
    convert.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    convert.add_argument(
        "--raster-upscale",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="Maximum automatic upscale for low-resolution PNG/JPG inputs",
    )
    convert.add_argument(
        "--max-page-megapixels",
        type=float,
        default=25.0,
        help="Reject pages larger than this decoded size; use 0 for no limit",
    )
    convert.add_argument(
        "--pixels-per-unit",
        type=float,
        default=1.0,
        help="Scale calibration; default is 1 pixel = 1 CAD unit",
    )
    convert.add_argument(
        "--no-auto-pdf-scale",
        action="store_true",
        help="Ignore a measurement scale embedded in the PDF",
    )
    convert.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Maximum preprocessing/vectorization passes",
    )
    convert.add_argument(
        "--target-score",
        type=float,
        default=92.0,
        help="Stop early when automatic QA reaches this percent",
    )
    convert.add_argument(
        "--ocr-languages",
        default="tha+eng",
        help="Tesseract language expression, such as tha+eng or eng",
    )
    convert.add_argument(
        "--text-font",
        default="Arial.ttf",
        help="TrueType font filename used for editable OCR text in CAD",
    )
    convert.add_argument("--no-ocr", action="store_true", help="Do not extract text")
    convert.add_argument(
        "--feedback-file",
        type=Path,
        default=Path("data/feedback.jsonl"),
        help="Local feedback store used for candidate ranking",
    )
    convert.add_argument("--zip", action="store_true", help="Also create a ZIP of all results")

    feedback = subparsers.add_parser(
        "feedback",
        help="Save a human quality score for a conversion report and retrain when possible.",
    )
    feedback.add_argument("report", type=Path, help="Generated *_report.json file")
    feedback.add_argument("--score", type=float, required=True, help="Quality score from 0 to 100")
    feedback.add_argument("--accept", action="store_true", help="Mark the output as accepted")
    feedback.add_argument("--note", default="", help="Optional correction note")
    feedback.add_argument(
        "--feedback-file",
        type=Path,
        default=Path("data/feedback.jsonl"),
        help="Local feedback store",
    )
    return parser


def run_convert(args: argparse.Namespace) -> int:
    config = ConversionConfig(
        pdf_dpi=args.dpi,
        max_page_pixels=(
            0
            if args.max_page_megapixels <= 0
            else int(args.max_page_megapixels * 1_000_000)
        ),
        pixels_per_unit=args.pixels_per_unit,
        auto_pdf_scale=not args.no_auto_pdf_scale,
        max_raster_upscale=args.raster_upscale,
        auto_mode=not args.manual,
        max_iterations=max(1, args.passes),
        desired_score=max(0.0, min(args.target_score / 100.0, 1.0)),
        ocr_enabled=not args.no_ocr,
        ocr_languages=args.ocr_languages,
        cad_text_font=args.text_font,
        export_dwg=args.dwg,
        oda_converter_path=args.oda_path,
    )
    converter = CADConverter(config=config, feedback_path=args.feedback_file)
    result = converter.convert(args.input, args.output)

    print(f"Completed {len(result.pages)} page(s) in: {result.output_directory}")
    for page in result.pages:
        metrics = page.candidate.metrics
        print(
            f"  page {page.page_number}: {page.dxf_path.name} | "
            f"QA {metrics.final_score * 100:.1f}% | "
            f"{len(page.candidate.entities)} editable entities | "
            f"source {page.profile.quality_score * 100:.1f}% | {page.stop_reason}"
        )
        for warning in page.warnings:
            print(f"    warning: {warning}")

    if args.zip:
        archive = create_output_archive(
            result.output_directory,
            result.output_directory.parent / f"{result.output_directory.name}.zip",
        )
        print(f"Archive: {archive}")
    return 0


def run_feedback(args: argparse.Namespace) -> int:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    candidate = report["best_candidate"]
    metrics = CandidateMetrics(**candidate["metrics"])
    learner = FeedbackLearner(args.feedback_file)
    summary = learner.record_feedback(
        source_name=str(report.get("source_name", args.report.name)),
        candidate_name=str(candidate.get("name", "unknown candidate")),
        metrics=metrics,
        score_percent=args.score,
        accepted=args.accept,
        note=args.note,
    )
    print(summary.message)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "convert":
        return run_convert(args)
    if args.command == "feedback":
        return run_feedback(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

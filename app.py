"""Local/Colab Gradio interface for AI CAD Converter."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from cad_converter import CADConverter, ConversionConfig
from cad_converter.archive import create_output_archive
from cad_converter.learning import FeedbackLearner
from cad_converter.models import CandidateMetrics


DEFAULT_FEEDBACK_PATH = Path(
    os.environ.get("AI_CAD_FEEDBACK_PATH", "data/feedback.jsonl")
)


def convert_uploaded_file(
    uploaded_file: str | None,
    auto_mode: bool,
    pixels_per_unit: float,
    auto_pdf_scale: bool,
    pdf_dpi: float,
    passes: float,
    target_score: float,
    ocr_languages: str,
    text_font: str,
    generate_dwg: bool,
    oda_path: str,
) -> tuple[list[tuple[str, str]], str, dict[str, Any], Any, Any]:
    if not uploaded_file:
        raise gr.Error("Please choose an image or PDF first.")

    job_root = Path(tempfile.mkdtemp(prefix="ai_cad_converter_"))
    output_dir = job_root / "output"
    config = ConversionConfig(
        pixels_per_unit=max(float(pixels_per_unit or 1.0), 0.000001),
        auto_pdf_scale=bool(auto_pdf_scale),
        pdf_dpi=max(72, int(pdf_dpi or 300)),
        auto_mode=bool(auto_mode),
        max_iterations=max(1, int(passes or 3)),
        desired_score=max(0.0, min(float(target_score or 92) / 100.0, 1.0)),
        ocr_languages=str(ocr_languages or "").strip() or "tha+eng",
        cad_text_font=str(text_font or "").strip() or "Arial.ttf",
        export_dwg=bool(generate_dwg),
        oda_converter_path=str(oda_path or "").strip() or None,
    )
    try:
        converter = CADConverter(config=config, feedback_path=DEFAULT_FEEDBACK_PATH)
        result = converter.convert(uploaded_file, output_dir)
    except Exception as exc:
        raise gr.Error(f"Conversion failed: {exc}") from exc

    archive_path = create_output_archive(
        output_dir,
        job_root / "AI_CAD_Converter_results.zip",
    )
    gallery: list[tuple[str, str]] = []
    summaries: list[str] = []
    report_paths: list[str] = []
    for page in result.pages:
        metric = page.candidate.metrics
        profile = page.profile
        gallery.append(
            (
                str(page.preview_path),
                (
                    f"Page {page.page_number} · auto QA {metric.final_score * 100:.1f}% "
                    f"· {len(page.candidate.entities)} editable entities"
                ),
            )
        )
        report_paths.append(str(page.report_path))
        outputs = [page.dxf_path.name]
        if page.dwg_path:
            outputs.append(page.dwg_path.name)
        summaries.append(
            f"- หน้า {page.page_number}: {', '.join(outputs)} · "
            f"QA {metric.final_score * 100:.1f}% · "
            f"คุณภาพไฟล์ต้นทาง {profile.quality_score * 100:.1f}% · "
            f"เลือก `{page.candidate.name}` · หยุดเพราะ `{page.stop_reason}`"
        )
        if profile.flags:
            summaries.append(f"  - AI ตรวจพบ: {', '.join(profile.flags)}")
        if page.decision_trace:
            tested = sum(int(step["tested_candidates"]) for step in page.decision_trace)
            summaries.append(
                f"  - ทดลอง {tested} candidate ใน {len(page.decision_trace)} รอบ"
            )
        summaries.extend(f"  - Notice: {warning}" for warning in page.warnings)

    description = "\n".join(
        [
            "## แปลงไฟล์เสร็จแล้ว",
            "",
            *summaries,
            "",
            "Preview จากซ้ายไปขวา: ต้นฉบับ, ภาพที่สร้างกลับจากวัตถุ CAD และ QA overlay. "
            "เขียว = ตรงกัน, แดง = ต้นฉบับที่ยังสร้างไม่ครบ, น้ำเงิน = CAD ที่เกินจากต้นฉบับ, "
            "กรอบม่วง = ข้อความ OCR",
            "",
            "คะแนนเป็นการเทียบภาพ ไม่ได้รับรอง scale, dimension หรือรายละเอียดวิศวกรรม "
            "จึงต้องตรวจแบบใน AutoCAD/GstarCAD ก่อนนำไปใช้งานจริง",
        ]
    )
    state: dict[str, Any] = {
        "report_paths": report_paths,
        "feedback_path": str(DEFAULT_FEEDBACK_PATH),
    }
    state["archive_path"] = str(archive_path)
    return gallery, description, state, gr.update(visible=True), gr.update(
        value=None,
        visible=False,
    )


def release_download(state: dict[str, Any] | None) -> tuple[Any, Any]:
    """Reveal the result archive only after the user has reviewed QA previews."""

    if not state or not state.get("archive_path"):
        raise gr.Error("Please run a conversion and review its preview first.")
    return (
        gr.update(value=state["archive_path"], visible=True),
        gr.update(visible=False),
    )


def save_feedback(
    state: dict[str, Any] | None,
    score: float,
    accepted: bool,
    note: str,
) -> str:
    if not state or not state.get("report_paths"):
        return "Run a conversion first, then rate the selected result."

    learner = FeedbackLearner(state.get("feedback_path", str(DEFAULT_FEEDBACK_PATH)))
    summaries: list[str] = []
    for report_path in state["report_paths"]:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        candidate = report["best_candidate"]
        metrics = CandidateMetrics(**candidate["metrics"])
        summary = learner.record_feedback(
            source_name=str(report.get("source_name", Path(report_path).name)),
            candidate_name=str(candidate.get("name", "unknown candidate")),
            metrics=metrics,
            score_percent=float(score),
            accepted=bool(accepted),
            note=note,
        )
        summaries.append(summary.message)
    return "\n".join(dict.fromkeys(summaries))


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Drawing to CAD") as interface:
        gr.Markdown(
            """
            # AI Drawing → CAD

            อัปโหลดรูปหรือ PDF แล้วระบบจะวิเคราะห์คุณภาพ เลือกวิธีปรับภาพด้วย OpenCV,
            อ่านข้อความด้วย Tesseract OCR และสร้าง LWPOLYLINE, CIRCLE, HATCH,
            TEXT หรือ MTEXT ที่แก้ไขได้ พร้อมวนตรวจ QA ก่อนดาวน์โหลด DXF/DWG
            """
        )
        conversion_state = gr.State(value={})
        with gr.Row():
            with gr.Column(scale=1):
                uploaded_file = gr.File(
                    label="ไฟล์รูปภาพหรือ PDF",
                    type="filepath",
                    file_types=["image", ".pdf"],
                )
                auto_mode = gr.Checkbox(
                    label="AI Auto mode — ให้ระบบเลือกวิธีประมวลผลและจำนวนรอบ",
                    value=True,
                )
                with gr.Accordion("ตั้งค่าขั้นสูง", open=False):
                    auto_pdf_scale = gr.Checkbox(
                        label="ใช้สเกลที่ฝังใน PDF อัตโนมัติ",
                        value=True,
                        info="แนะนำสำหรับ PDF ที่ส่งออกจาก CAD เพื่อให้ขนาดตรงต้นฉบับ",
                    )
                    pixels_per_unit = gr.Number(
                        label="Pixels ต่อ 1 หน่วย CAD",
                        value=1.0,
                        minimum=0.000001,
                        info="ใช้ค่านี้เมื่อ PDF ไม่มีสเกลฝังอยู่ หรือปิดสเกลอัตโนมัติ",
                    )
                    pdf_dpi = gr.Slider(
                        label="ความละเอียด PDF (DPI)",
                        minimum=150,
                        maximum=600,
                        value=300,
                        step=25,
                    )
                    passes = gr.Slider(
                        label="จำนวนรอบ QA สูงสุด",
                        minimum=1,
                        maximum=3,
                        value=3,
                        step=1,
                    )
                    target_score = gr.Slider(
                        label="หยุดเมื่อคะแนน QA ถึง (%)",
                        minimum=70,
                        maximum=99,
                        value=92,
                        step=1,
                    )
                    ocr_languages = gr.Textbox(
                        label="ภาษา Tesseract OCR",
                        value="tha+eng",
                        info="ใช้ tha+eng เมื่อเครื่องติดตั้งชุดภาษาไทยและอังกฤษแล้ว",
                    )
                    text_font = gr.Textbox(
                        label="ฟอนต์ CAD สำหรับข้อความ OCR",
                        value="Arial.ttf",
                        info="เลือกฟอนต์ที่รองรับไทย เช่น Arial.ttf",
                    )
                    generate_dwg = gr.Checkbox(
                        label="สร้าง DWG เพิ่มด้วย ODA File Converter ในเครื่อง",
                        value=False,
                    )
                    oda_path = gr.Textbox(
                        label="ตำแหน่ง ODAFileConverter (ไม่บังคับ)",
                        placeholder="Example: C:/Program Files/ODA/ODAFileConverter.exe",
                    )
                convert_button = gr.Button(
                    "แปลงเป็น CAD และดู Preview",
                    variant="primary",
                )

            with gr.Column(scale=1):
                download = gr.File(
                    label="ดาวน์โหลดผลลัพธ์ (DXF/DWG + QA report)",
                    visible=False,
                )
                preview = gr.Gallery(
                    label="Preview ก่อนดาวน์โหลด",
                    columns=1,
                    object_fit="contain",
                    height="auto",
                )
                conversion_summary = gr.Markdown()
                confirm_preview_button = gr.Button(
                    "ยืนยัน Preview แล้วแสดงไฟล์ดาวน์โหลด",
                    variant="primary",
                    visible=False,
                )

        gr.Markdown("## สอนระบบให้เลือกผลที่ดีขึ้นในครั้งต่อไป")
        with gr.Row():
            feedback_score = gr.Slider(
                label="คะแนนคุณภาพจากคุณ",
                minimum=0,
                maximum=100,
                value=85,
                step=1,
            )
            accepted = gr.Checkbox(label="ยอมรับผลลัพธ์นี้", value=True)
        feedback_note = gr.Textbox(
            label="ครั้งต่อไปควรปรับอะไร",
            placeholder="ตัวอย่าง: เส้นสีแดงบางเส้นหาย หรือข้อความไทยต้องแม่นขึ้น",
            lines=2,
        )
        feedback_button = gr.Button("บันทึก feedback และเรียนรู้")
        feedback_status = gr.Markdown()

        convert_button.click(
            convert_uploaded_file,
            inputs=[
                uploaded_file,
                auto_mode,
                pixels_per_unit,
                auto_pdf_scale,
                pdf_dpi,
                passes,
                target_score,
                ocr_languages,
                text_font,
                generate_dwg,
                oda_path,
            ],
            outputs=[
                preview,
                conversion_summary,
                conversion_state,
                confirm_preview_button,
                download,
            ],
        )
        confirm_preview_button.click(
            release_download,
            inputs=[conversion_state],
            outputs=[download, confirm_preview_button],
        )
        feedback_button.click(
            save_feedback,
            inputs=[conversion_state, feedback_score, accepted, feedback_note],
            outputs=feedback_status,
        )
    return interface.queue(max_size=8, default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the AI CAD Converter UI.")
    parser.add_argument("--share", action="store_true", help="Ask Gradio for a share link")
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Open the local UI in the default web browser",
    )
    parser.add_argument("--server-name", default=None, help="Optional bind address")
    parser.add_argument("--server-port", type=int, default=None, help="Optional port")
    args = parser.parse_args()
    build_app().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=args.inbrowser,
        max_threads=4,
        max_file_size=os.environ.get("AI_CAD_MAX_UPLOAD", "50mb"),
    )


if __name__ == "__main__":
    main()

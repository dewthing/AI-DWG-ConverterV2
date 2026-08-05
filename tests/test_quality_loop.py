from __future__ import annotations

import json

import cv2
import numpy as np

from cad_converter import CADConverter, ConversionConfig


def test_auto_mode_forces_blurred_input_through_recovery_passes(tmp_path):
    image = np.full((280, 420, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 35), (385, 240), (0, 0, 0), 2)
    cv2.line(image, (35, 130), (385, 130), (0, 0, 0), 2)
    image = cv2.GaussianBlur(image, (13, 13), 4)
    source = tmp_path / "blurred.png"
    cv2.imwrite(str(source), image)

    converter = CADConverter(
        ConversionConfig(
            auto_mode=True,
            ocr_enabled=False,
            max_iterations=3,
            desired_score=0.0,
            min_line_length=16,
            contour_min_area=60,
            auto_upscale_low_resolution=False,
        ),
        feedback_path=tmp_path / "feedback.jsonl",
    )
    result = converter.convert(source, tmp_path / "output")
    page = result.pages[0]
    report = json.loads(page.report_path.read_text(encoding="utf-8"))

    assert page.profile.recommended_iterations == 3
    assert len(page.decision_trace) == 3
    assert page.stop_reason == "target_score_reached"
    assert report["decision_trace"] == page.decision_trace
    assert report["stop_reason"] == "target_score_reached"
    assert report["input_profile"]["flags"]

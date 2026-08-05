from __future__ import annotations

import cv2
import numpy as np

from cad_converter.decision import AutoDecisionEngine
from cad_converter.preprocessing import generate_variants


def _sample_drawing() -> np.ndarray:
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.line(image, (30, 50), (570, 50), (0, 0, 0), 2)
    cv2.rectangle(image, (100, 100), (500, 300), (0, 0, 0), 2)
    cv2.putText(
        image,
        "ATC-01",
        (180, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )
    return image


def test_blurred_page_requests_full_recovery_loop():
    engine = AutoDecisionEngine()
    blurred = cv2.GaussianBlur(_sample_drawing(), (13, 13), 4)

    profile = engine.analyse(blurred)
    final_decision = engine.decide_iteration(profile, 2)

    assert "blurred" in profile.flags
    assert "low_contrast" in profile.flags
    assert profile.recommended_iterations == 3
    assert "unsharp_otsu" in final_decision.strategies
    assert "blackhat_ink" in final_decision.strategies


def test_strategy_filter_only_returns_auto_selected_candidates():
    selected = ["background_normalized", "bilateral_adaptive"]
    variants = generate_variants(_sample_drawing(), iteration=1, selected_names=selected)

    assert variants
    assert {variant.name for variant in variants}.issubset(set(selected))

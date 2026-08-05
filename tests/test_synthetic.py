from __future__ import annotations

import cv2
import ezdxf
import numpy as np

from cad_converter import CADConverter, ConversionConfig
from cad_converter.preprocessing import reference_ink_mask


def test_synthetic_drawing_exports_editable_dxf(tmp_path):
    image = np.full((320, 480, 3), 255, dtype=np.uint8)
    cv2.line(image, (40, 40), (430, 40), (0, 0, 0), 2)
    cv2.line(image, (40, 40), (40, 260), (0, 0, 0), 2)
    cv2.rectangle(image, (130, 120), (330, 230), (0, 0, 0), 2)
    cv2.circle(image, (390, 190), 38, (0, 0, 0), 2)

    source = tmp_path / "sample.png"
    cv2.imwrite(str(source), image)
    converter = CADConverter(
        ConversionConfig(
            ocr_enabled=False,
            max_iterations=1,
            min_line_length=20,
            contour_min_area=80,
        ),
        feedback_path=tmp_path / "feedback.jsonl",
    )
    result = converter.convert(source, tmp_path / "output")

    dxf_path = result.pages[0].dxf_path
    circles = [
        entity
        for entity in result.pages[0].candidate.entities
        if entity.kind == "CIRCLE"
    ]
    document = ezdxf.readfile(dxf_path)
    entity_types = {entity.dxftype() for entity in document.modelspace()}
    assert dxf_path.exists()
    assert "LINE" in entity_types or "LWPOLYLINE" in entity_types
    assert 1 <= len(circles) <= 2
    assert any(
        circle.center is not None
        and abs(circle.center[0] - 390) <= 4
        and abs(circle.center[1] - 190) <= 4
        for circle in circles
    )
    assert result.pages[0].preview_path.exists()


def test_reference_mask_keeps_pale_coloured_cad_layers():
    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 20), (150, 20), (120, 230, 170), 2)
    cv2.line(image, (10, 50), (150, 50), (180, 190, 245), 2)
    mask = reference_ink_mask(image)

    assert np.count_nonzero(mask[18:23, 10:151]) > 200
    assert np.count_nonzero(mask[48:53, 10:151]) > 200

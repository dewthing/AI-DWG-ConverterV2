from __future__ import annotations

import json

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
            auto_upscale_low_resolution=False,
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


def test_dense_tables_do_not_become_hundreds_of_false_circles():
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    for origin_x, origin_y, table_width, table_height, rows, columns in (
        (80, 120, 260, 180, 12, 8),
        (430, 250, 260, 220, 14, 7),
        (270, 400, 180, 120, 8, 5),
    ):
        cv2.rectangle(
            image,
            (origin_x, origin_y),
            (origin_x + table_width, origin_y + table_height),
            (0, 0, 0),
            1,
        )
        for row in range(1, rows):
            y = origin_y + row * table_height // rows
            cv2.line(
                image,
                (origin_x, y),
                (origin_x + table_width, y),
                (0, 0, 0),
                1,
            )
        for column in range(1, columns):
            x = origin_x + column * table_width // columns
            cv2.line(
                image,
                (x, origin_y),
                (x, origin_y + table_height),
                (0, 0, 0),
                1,
            )
        for row in range(rows):
            cv2.putText(
                image,
                "NC1",
                (origin_x + 3, origin_y + (row + 1) * table_height // rows - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    vectorizer = CADConverter(
        ConversionConfig(ocr_enabled=False, max_iterations=1),
    ).vectorizer
    entities = vectorizer.vectorize(reference_ink_mask(image))
    circles = [entity for entity in entities if entity.kind == "CIRCLE"]

    assert circles == []


def test_low_resolution_raster_is_upscaled_without_changing_cad_extents(tmp_path):
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 40), (450, 260), (0, 0, 0), 2)
    source = tmp_path / "small.png"
    cv2.imwrite(str(source), image)

    converter = CADConverter(
        ConversionConfig(
            ocr_enabled=False,
            max_iterations=1,
            low_resolution_target_long_edge=1200,
            max_raster_upscale=3,
        ),
        feedback_path=tmp_path / "feedback.jsonl",
    )
    result = converter.convert(source, tmp_path / "upscaled_output")
    page = result.pages[0]
    report = json.loads(page.report_path.read_text(encoding="utf-8"))
    drawing = ezdxf.readfile(page.dxf_path)
    points = [
        point
        for entity in drawing.modelspace().query("LINE LWPOLYLINE")
        for point in (
            list(entity.get_points())
            if entity.dxftype() == "LWPOLYLINE"
            else [entity.dxf.start, entity.dxf.end]
        )
    ]
    x_values = [float(point[0]) for point in points]

    assert report["processing_scale"] == 3
    assert report["source_image_size"] == {"width": 500, "height": 300}
    assert report["image_size"] == {"width": 500, "height": 300}
    assert report["processing_image_size"] == {"width": 1500, "height": 900}
    assert report["effective_export_settings"]["pixels_per_unit"] == 3.0
    assert max(x_values) <= 500.0

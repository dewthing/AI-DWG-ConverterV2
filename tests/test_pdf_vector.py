from __future__ import annotations

import fitz
import ezdxf
import json
import pytest

from cad_converter import CADConverter, ConversionConfig
from cad_converter.pdf_vector import extract_native_pdf_vectors


def test_native_pdf_paths_and_text_are_extracted(tmp_path):
    source = tmp_path / "vector_source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    shape = page.new_shape()
    shape.draw_line((10, 10), (80, 10))
    shape.draw_rect((20, 20, 60, 50))
    shape.finish(color=(0, 0, 0))
    shape.commit()
    page.insert_text((10, 80), "Hello", fontsize=12)
    document.save(source)
    document.close()

    pages = extract_native_pdf_vectors(source, 144)
    assert len(pages) == 1
    assert any(entity.kind == "LWPOLYLINE" for entity in pages[0].geometry_entities)
    assert any(entity.kind == "LWPOLYLINE" for entity in pages[0].geometry_entities)
    assert pages[0].text_items[0].text == "Hello"


def test_rotated_pdf_vectors_and_text_match_rendered_page_coordinates(tmp_path):
    source = tmp_path / "rotated_vector_source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    shape = page.new_shape()
    shape.draw_line((10, 20), (80, 20))
    shape.finish(color=(1, 0, 0))
    shape.commit()
    page.insert_text((10, 80), "Rotated", fontsize=12, color=(0, 0, 1))
    page.set_rotation(270)
    document.save(source)
    document.close()

    extracted = extract_native_pdf_vectors(source, 72)[0]
    line = next(
        entity
        for entity in extracted.geometry_entities
        if entity.kind == "LWPOLYLINE" and len(entity.points) == 2
    )
    assert [point[0] for point in line.points] == pytest.approx([20.0, 20.0])
    assert sorted(point[1] for point in line.points) == pytest.approx([120.0, 190.0])
    assert line.extra["true_color"] == 0xFF0000

    text = extracted.text_items[0]
    assert text.text == "Rotated"
    assert text.rotation == pytest.approx(90.0)
    assert text.color == 0x0000FF
    assert 0 <= text.bbox[0] < 100
    assert 0 <= text.bbox[1] < 200


def test_largest_pdf_viewport_measurement_is_used_for_cad_scale(tmp_path):
    source = tmp_path / "measured_source.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    viewports = """
    [
      << /Type /Viewport /BBox [ 0 0 200 100 ]
         /Measure << /Type /Measure /Subtype /RL
                     /X [ << /C .0112 /D 1 /U (m) >> ] >> >>
      << /Type /Viewport /BBox [ 20 20 40 40 ]
         /Measure << /Type /Measure /Subtype /RL
                     /X [ << /C 27.275 /D 1 /U (m) >> ] >> >>
    ]
    """
    document.xref_set_key(page.xref, "VP", viewports)
    document.save(source)
    document.close()

    extracted = extract_native_pdf_vectors(source, 150)[0]
    assert extracted.cad_units_per_point == pytest.approx(0.0112)


def test_embedded_pdf_scale_is_applied_to_exported_dxf(tmp_path):
    source = tmp_path / "measured_drawing.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    for y in range(10, 70, 5):
        shape = page.new_shape()
        shape.draw_line((10, y), (190, y))
        shape.finish(color=(0, 0, 0))
        shape.commit()
    document.xref_set_key(
        page.xref,
        "VP",
        "[ << /Type /Viewport /BBox [ 0 0 200 100 ] "
        "/Measure << /Type /Measure /Subtype /RL "
        "/X [ << /C .01 /D 1 /U (m) >> ] >> >> ]",
    )
    document.save(source)
    document.close()

    converter = CADConverter(
        ConversionConfig(pdf_dpi=72, max_iterations=1, ocr_enabled=False),
        feedback_path=tmp_path / "feedback.jsonl",
    )
    result = converter.convert(source, tmp_path / "output")
    drawing = ezdxf.readfile(result.pages[0].dxf_path)
    x_coordinates = [
        float(point[0])
        for entity in drawing.modelspace().query("LWPOLYLINE")
        for point in entity.get_points()
    ]
    report = json.loads(result.pages[0].report_path.read_text(encoding="utf-8"))

    assert min(x_coordinates) == pytest.approx(0.1, abs=0.01)
    assert max(x_coordinates) == pytest.approx(1.9, abs=0.01)
    assert report["effective_export_settings"]["pixels_per_unit"] == pytest.approx(100.0)

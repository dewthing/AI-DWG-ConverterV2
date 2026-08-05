from __future__ import annotations

import fitz

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
    assert any(entity.kind == "LINE" for entity in pages[0].geometry_entities)
    assert any(entity.kind == "LWPOLYLINE" for entity in pages[0].geometry_entities)
    assert pages[0].text_items[0].text == "Hello"


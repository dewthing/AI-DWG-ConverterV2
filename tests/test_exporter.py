from __future__ import annotations

import ezdxf

from cad_converter.exporter import export_dxf
from cad_converter.models import CadEntity
from cad_converter.settings import ConversionConfig


def test_pdf_hatch_and_native_text_export_as_editable_entities(tmp_path):
    entities = [
        CadEntity(
            kind="HATCH",
            layer="PDF_Solid Fills",
            points=[(10, 10), (60, 10), (60, 40), (10, 40)],
            closed=True,
            extra={"true_color": 0xFF0000},
        ),
        CadEntity(
            kind="TEXT",
            layer="PDF_Text",
            text="Editable",
            bbox=(20, 50, 80, 12),
            height=10,
            extra={"native_pdf_text": True, "rotation": 15.0},
        ),
    ]
    target = export_dxf(
        entities,
        (100, 100),
        tmp_path / "editable.dxf",
        ConversionConfig(pixels_per_unit=10.0),
    )

    document = ezdxf.readfile(target)
    modelspace = document.modelspace()
    assert [entity.dxftype() for entity in modelspace] == ["HATCH", "MTEXT"]
    assert modelspace[0].dxf.layer == "PDF_Solid Fills"
    assert modelspace[0].dxf.true_color == 0xFF0000
    assert modelspace[1].dxf.layer == "PDF_Text"
    assert modelspace[1].dxf.char_height == 1.0

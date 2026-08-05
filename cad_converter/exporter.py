"""Editable DXF export and optional local DXF-to-DWG conversion."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import ezdxf

from .models import CadEntity
from .settings import ConversionConfig


@dataclass(slots=True)
class DWGExportResult:
    path: Path | None
    warning: str | None = None


def export_dxf(
    entities: list[CadEntity],
    image_shape: tuple[int, int],
    output_path: str | Path,
    config: ConversionConfig,
) -> Path:
    """Write real CAD entities in a Unicode-capable DXF R2018 drawing."""

    height, _ = image_shape
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    document = ezdxf.new("R2018", setup=True)
    document.header["$INSUNITS"] = 0  # Unitless; source scale is retained exactly.
    _add_layers(document, config)
    modelspace = document.modelspace()

    for entity in entities:
        if entity.kind == "LINE" and entity.start and entity.end:
            modelspace.add_line(
                _to_cad_point(entity.start, height, config.pixels_per_unit),
                _to_cad_point(entity.end, height, config.pixels_per_unit),
                dxfattribs={"layer": entity.layer},
            )
        elif entity.kind == "CIRCLE" and entity.center and entity.radius is not None:
            modelspace.add_circle(
                _to_cad_point(entity.center, height, config.pixels_per_unit),
                entity.radius / config.pixels_per_unit,
                dxfattribs={"layer": entity.layer},
            )
        elif (
            entity.kind == "ARC"
            and entity.center
            and entity.radius is not None
            and entity.start_angle is not None
            and entity.end_angle is not None
        ):
            modelspace.add_arc(
                _to_cad_point(entity.center, height, config.pixels_per_unit),
                entity.radius / config.pixels_per_unit,
                entity.start_angle,
                entity.end_angle,
                dxfattribs={"layer": entity.layer},
            )
        elif entity.kind == "LWPOLYLINE" and entity.points:
            modelspace.add_lwpolyline(
                [
                    _to_cad_point(point, height, config.pixels_per_unit)
                    for point in entity.points
                ],
                close=entity.closed,
                dxfattribs={"layer": entity.layer},
            )
        elif entity.kind == "TEXT" and entity.text and entity.bbox:
            x, y, _, text_height_pixels = entity.bbox
            insertion = _to_cad_point(
                (x, y + text_height_pixels),
                height,
                config.pixels_per_unit,
            )
            modelspace.add_text(
                entity.text,
                dxfattribs={
                    "layer": entity.layer,
                    "style": config.cad_text_style,
                    "height": max(
                        1.0,
                        (entity.height or text_height_pixels)
                        / config.pixels_per_unit,
                    ),
                    "insert": insertion,
                },
            )
    document.saveas(target)
    return target


def export_dwg_with_oda(
    dxf_path: str | Path,
    output_path: str | Path,
    config: ConversionConfig,
) -> DWGExportResult:
    """Convert a generated DXF to DWG with a locally installed ODA converter.

    DWG is a proprietary format. This program therefore writes standards-based
    editable DXF itself and invokes a local, user-installed converter only when
    an actual DWG file is requested.
    """

    executable = _find_oda_converter(config.oda_converter_path)
    if executable is None:
        return DWGExportResult(
            path=None,
            warning=(
                "DWG was not produced because ODA File Converter was not found. "
                "The editable DXF output is ready to open in AutoCAD/GstarCAD, "
                "or configure the ODA converter path and run again."
            ),
        )

    source_dxf = Path(dxf_path).resolve()
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ai_cad_oda_") as temporary:
        source_dir = Path(temporary) / "source"
        converter_output = Path(temporary) / "output"
        source_dir.mkdir()
        converter_output.mkdir()
        staged = source_dir / source_dxf.name
        shutil.copy2(source_dxf, staged)

        command = [
            str(executable),
            str(source_dir),
            str(converter_output),
            "*.dxf",
            config.dwg_version,
            "DWG",
            "0",
            "1",
        ]
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        possible = list(converter_output.rglob("*.dwg"))
        if process.returncode == 0 and possible:
            shutil.copy2(possible[0], destination)
            return DWGExportResult(path=destination)

        output = (process.stderr or process.stdout or "").strip().replace("\n", " ")
        message = output[:500] if output else f"exit code {process.returncode}"
        return DWGExportResult(
            path=None,
            warning=f"ODA File Converter did not create DWG ({message}).",
        )


def _add_layers(document: ezdxf.document.Drawing, config: ConversionConfig) -> None:
    layers = {
        "GEOMETRY": 7,
        "SYMBOLS": 4,
        "TEXT": 2,
        "PDF_VECTOR": 3,
    }
    for name, colour in layers.items():
        if not document.layers.has_entry(name):
            document.layers.new(name, dxfattribs={"color": colour})
    if not document.styles.has_entry(config.cad_text_style):
        document.styles.new(
            config.cad_text_style,
            dxfattribs={"font": config.cad_text_font},
        )


def _to_cad_point(
    point: tuple[float, float],
    image_height: int,
    pixels_per_unit: float,
) -> tuple[float, float]:
    scale = max(pixels_per_unit, 1e-9)
    return (point[0] / scale, (image_height - point[1]) / scale)


def _find_oda_converter(configured_path: str | None) -> Path | None:
    candidates: list[str] = []
    if configured_path:
        candidates.append(configured_path)
    environment_path = os.environ.get("ODA_FILE_CONVERTER")
    if environment_path:
        candidates.append(environment_path)
    candidates.extend(
        [
            "ODAFileConverter",
            "ODAFileConverter.exe",
            "ODAFileConverter.app",
        ]
    )

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
        discovered = shutil.which(candidate)
        if discovered:
            return Path(discovered).resolve()
    return None

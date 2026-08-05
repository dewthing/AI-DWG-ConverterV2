"""Extract native vector geometry and text from born-digital PDF pages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import atan2, degrees, hypot, isfinite
from pathlib import Path
import re
from typing import Any

import fitz

from .models import CadEntity, OCRItem


@dataclass(slots=True)
class PDFVectorPage:
    geometry_entities: list[CadEntity] = field(default_factory=list)
    text_items: list[OCRItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cad_units_per_point: float | None = None


def extract_native_pdf_vectors(path: str | Path, dpi: int) -> list[PDFVectorPage]:
    """Extract PDF paths and selectable text in the same pixel grid as rendering.

    This path is used opportunistically: a born-digital PDF retains its original
    vector linework and selectable text, whereas a scanned PDF naturally falls
    back to the OpenCV + OCR pipeline.
    """

    source = Path(path)
    scale = max(dpi, 72) / 72.0
    results: list[PDFVectorPage] = []
    with fitz.open(source) as document:
        for page in document:
            result = PDFVectorPage(
                cad_units_per_point=_page_measure_scale(document, page),
            )
            try:
                rotation_matrix = page.rotation_matrix
                for drawing in page.get_drawings():
                    result.geometry_entities.extend(
                        _drawing_entities(drawing, scale, rotation_matrix)
                    )
                result.geometry_entities = _merge_touching_hatches(
                    result.geometry_entities,
                    tolerance=max(0.05, scale * 0.03),
                )
                result.geometry_entities = _join_pdf_paths(
                    result.geometry_entities,
                    tolerance=max(0.05, scale * 0.05),
                )
                result.text_items.extend(
                    _page_text_items(page, scale, rotation_matrix)
                )
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                result.warnings.append(
                    f"Native PDF vector extraction was skipped: {str(exc).splitlines()[0]}"
                )
                result.geometry_entities.clear()
                result.text_items.clear()
            results.append(result)
    return results


def _drawing_entities(
    drawing: dict[str, Any],
    scale: float,
    transform: fitz.Matrix | None = None,
) -> list[CadEntity]:
    entities: list[CadEntity] = []
    stroke_extra = _drawing_extra(drawing, use_fill=False, scale=scale)
    fill_extra = _drawing_extra(drawing, use_fill=True, scale=scale)
    groups = _drawing_path_groups(drawing, scale, transform)

    fill_paths = [
        _without_duplicate_endpoint(points)
        for points, _commands, closed in groups
        if closed and len(_without_duplicate_endpoint(points)) >= 3
    ]
    if (
        drawing.get("fill") is not None
        and float(drawing.get("fill_opacity", 1.0) or 0.0) > 0.0
        and fill_paths
    ):
        entities.append(
            CadEntity(
                kind="HATCH",
                layer="PDF_Solid Fills",
                points=fill_paths[0],
                boundary_paths=fill_paths,
                closed=True,
                confidence=1.0,
                extra=fill_extra.copy(),
            )
        )
        # PDFIMPORT represents a painted solid primarily as a HATCH / SOLID.
        # Keeping its stroke paths as separate polylines duplicates the same
        # boundaries and creates hundreds of tiny, overlapping CAD entities.
        return entities

    for points, commands, closed in groups:
        if len(points) < 2:
            continue
        path_points = _without_duplicate_endpoint(points)
        circle = _circle_from_path(points, commands, closed)
        if circle is not None:
            center, radius = circle
            entities.append(
                CadEntity(
                    kind="CIRCLE",
                    layer="PDF_Geometry",
                    center=center,
                    radius=radius,
                    confidence=1.0,
                    extra=stroke_extra.copy(),
                )
            )
        elif all(command == "l" for command in commands) and len(path_points) == 2:
            entities.append(
                CadEntity(
                    kind="LINE",
                    layer="PDF_Geometry",
                    start=path_points[0],
                    end=path_points[1],
                    confidence=1.0,
                    extra=stroke_extra.copy(),
                )
            )
        else:
            entities.append(
                CadEntity(
                    kind="LWPOLYLINE",
                    layer="PDF_Geometry",
                    points=path_points,
                    closed=closed,
                    confidence=0.98 if "c" in commands else 1.0,
                    extra=stroke_extra.copy(),
                )
            )
    return entities


def _drawing_path_groups(
    drawing: dict[str, Any],
    scale: float,
    transform: fitz.Matrix | None,
) -> list[tuple[list[tuple[float, float]], list[str], bool]]:
    groups: list[tuple[list[tuple[float, float]], list[str], bool]] = []
    current_points: list[tuple[float, float]] = []
    current_commands: list[str] = []
    tolerance = max(0.05, scale * 0.02)

    def flush(force_closed: bool = False) -> None:
        nonlocal current_points, current_commands
        if current_points:
            closed = force_closed or _points_close(
                current_points[0],
                current_points[-1],
                tolerance,
            )
            groups.append((current_points, current_commands, closed))
        current_points = []
        current_commands = []

    for item in drawing.get("items", []):
        if not item:
            continue
        command = item[0]
        if command == "l" and len(item) >= 3:
            segment = [
                _scaled_point(item[1], scale, transform),
                _scaled_point(item[2], scale, transform),
            ]
        elif command == "c" and len(item) >= 5:
            segment = _sample_cubic_bezier(
                item[1],
                item[2],
                item[3],
                item[4],
                scale,
                transform,
            )
        elif command == "re" and len(item) >= 2:
            flush()
            rectangle = item[1]
            points = _transformed_corners(
                (
                    (rectangle.x0, rectangle.y0),
                    (rectangle.x1, rectangle.y0),
                    (rectangle.x1, rectangle.y1),
                    (rectangle.x0, rectangle.y1),
                ),
                scale,
                transform,
            )
            groups.append((points, ["re"], True))
            continue
        elif command == "qu" and len(item) >= 2:
            flush()
            quad = item[1]
            points = _transformed_corners(
                (
                    (quad.ul.x, quad.ul.y),
                    (quad.ur.x, quad.ur.y),
                    (quad.lr.x, quad.lr.y),
                    (quad.ll.x, quad.ll.y),
                ),
                scale,
                transform,
            )
            groups.append((points, ["qu"], True))
            continue
        else:
            flush()
            continue

        if current_points and _points_close(
            current_points[-1],
            segment[0],
            tolerance,
        ):
            current_points.extend(segment[1:])
        elif current_points and _points_close(
            current_points[-1],
            segment[-1],
            tolerance,
        ):
            current_points.extend(reversed(segment[:-1]))
        else:
            flush()
            current_points = segment
        current_commands.append(command)

    flush(bool(drawing.get("closePath")))
    return groups


def _page_text_items(
    page: fitz.Page,
    scale: float,
    transform: fitz.Matrix | None = None,
) -> list[OCRItem]:
    items: list[OCRItem] = []
    text = page.get_text("dict")
    for block in text.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            rotation = _cad_rotation(line.get("dir", (1.0, 0.0)), transform)
            for span in line.get("spans", []):
                value = str(span.get("text", "")).strip()
                bbox = span.get("bbox")
                if not value or not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = (float(value) for value in bbox)
                corners = _transformed_corners(
                    ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                    scale,
                    transform,
                )
                xs = [point[0] for point in corners]
                ys = [point[1] for point in corners]
                left, top = min(xs), min(ys)
                right, bottom = max(xs), max(ys)
                items.append(
                    OCRItem(
                        text=value,
                        confidence=1.0,
                        bbox=(
                            int(round(left)),
                            int(round(top)),
                            max(1, int(round(right - left))),
                            max(1, int(round(bottom - top))),
                        ),
                        rotation=rotation,
                        font=str(span.get("font", "")).strip() or None,
                        color=_span_color(span.get("color")),
                    )
                )
    return items


def _scaled_point(
    point: Any,
    scale: float,
    transform: fitz.Matrix | None = None,
) -> tuple[float, float]:
    transformed = fitz.Point(float(point.x), float(point.y))
    if transform is not None:
        transformed = transformed * transform
    return (float(transformed.x) * scale, float(transformed.y) * scale)


def _sample_cubic_bezier(
    point0: Any,
    point1: Any,
    point2: Any,
    point3: Any,
    scale: float,
    transform: fitz.Matrix | None = None,
    steps: int = 16,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = index / steps
        inverse = 1.0 - t
        x = (
            inverse**3 * point0.x
            + 3 * inverse**2 * t * point1.x
            + 3 * inverse * t**2 * point2.x
            + t**3 * point3.x
        )
        y = (
            inverse**3 * point0.y
            + 3 * inverse**2 * t * point1.y
            + 3 * inverse * t**2 * point2.y
            + t**3 * point3.y
        )
        result.append(_scaled_point(fitz.Point(x, y), scale, transform))
    return result


def _transformed_corners(
    points: tuple[tuple[float, float], ...],
    scale: float,
    transform: fitz.Matrix | None,
) -> list[tuple[float, float]]:
    return [
        _scaled_point(fitz.Point(x, y), scale, transform)
        for x, y in points
    ]


def _cad_rotation(
    direction: Any,
    transform: fitz.Matrix | None,
) -> float:
    try:
        dx, dy = (float(direction[0]), float(direction[1]))
    except (TypeError, ValueError, IndexError):
        return 0.0
    if transform is not None:
        origin = fitz.Point(0.0, 0.0) * transform
        endpoint = fitz.Point(dx, dy) * transform
        dx = endpoint.x - origin.x
        dy = endpoint.y - origin.y
    return float(degrees(atan2(-dy, dx)) % 360.0)


def _page_measure_scale(
    document: fitz.Document,
    page: fitz.Page,
) -> float | None:
    """Return embedded real-world CAD units per PDF point, when available.

    Engineering PDFs can contain more than one viewport measurement. AutoCAD's
    PDFIMPORT uses the page-scale viewport, so prefer the positive scale attached
    to the largest viewport bounding box.
    """

    try:
        page_object = document.xref_object(page.xref, compressed=False)
    except (RuntimeError, ValueError):
        return None

    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
    bbox_pattern = re.compile(
        rf"/BBox\s*\[\s*({number})\s+({number})\s+({number})\s+({number})\s*\]",
        re.DOTALL,
    )
    scale_pattern = re.compile(
        rf"/X\s*\[\s*<<.*?/C\s+({number})(?:\s|/|>>)",
        re.DOTALL,
    )
    boxes = list(bbox_pattern.finditer(page_object))
    candidates: list[tuple[float, float]] = []
    for index, match in enumerate(boxes):
        segment_end = boxes[index + 1].start() if index + 1 < len(boxes) else len(page_object)
        segment = page_object[match.end() : segment_end]
        if "/Measure" not in segment:
            continue
        scale_match = scale_pattern.search(segment)
        if scale_match is None:
            continue
        try:
            x0, y0, x1, y1 = (float(match.group(item)) for item in range(1, 5))
            value = float(scale_match.group(1))
        except (TypeError, ValueError):
            continue
        area = abs((x1 - x0) * (y1 - y0))
        if area > 0.0 and value > 0.0 and isfinite(value):
            candidates.append((area, value))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    fallback = scale_pattern.search(page_object)
    if fallback is not None:
        try:
            value = float(fallback.group(1))
        except (TypeError, ValueError):
            return None
        if value > 0.0 and isfinite(value):
            return value
    return None


def _drawing_extra(
    drawing: dict[str, Any],
    *,
    use_fill: bool,
    scale: float,
) -> dict[str, Any]:
    colour = drawing.get("fill" if use_fill else "color")
    extra: dict[str, Any] = {}
    source_layer = str(drawing.get("layer", "")).strip()
    if source_layer:
        extra["source_pdf_layer"] = source_layer
    if not use_fill:
        try:
            line_width = float(drawing.get("width", 0.0)) * scale
        except (TypeError, ValueError):
            line_width = 0.0
        if line_width > 0.0:
            extra["line_width_pixels"] = line_width

    true_colour = _colour_tuple_to_int(colour)
    if true_colour is None:
        return extra
    if use_fill:
        extra["source_is_white"] = true_colour == 0xFFFFFF
    if true_colour in {0x000000, 0xFFFFFF}:
        extra["aci_color"] = 7
    else:
        extra["true_color"] = true_colour
    return extra


def _points_close(
    first: tuple[float, float],
    second: tuple[float, float],
    tolerance: float,
) -> bool:
    return hypot(first[0] - second[0], first[1] - second[1]) <= tolerance


def _without_duplicate_endpoint(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(points) > 2 and _points_close(points[0], points[-1], 1e-4):
        return points[:-1]
    return points.copy()


def _circle_from_path(
    points: list[tuple[float, float]],
    commands: list[str],
    closed: bool,
) -> tuple[tuple[float, float], float] | None:
    """Recognise the standard four-cubic representation of a PDF circle."""

    if not closed or len(commands) != 4 or any(command != "c" for command in commands):
        return None
    if len(points) < 16:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    radius_x = (max(xs) - min(xs)) / 2.0
    radius_y = (max(ys) - min(ys)) / 2.0
    radius = (radius_x + radius_y) / 2.0
    if radius <= 1e-6 or abs(radius_x - radius_y) / radius > 0.02:
        return None
    center = ((max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0)
    radial_errors = [
        abs(hypot(point[0] - center[0], point[1] - center[1]) - radius)
        for point in points
    ]
    if max(radial_errors, default=0.0) > radius * 0.025:
        return None
    return center, radius


def _merge_touching_hatches(
    entities: list[CadEntity],
    *,
    tolerance: float,
) -> list[CadEntity]:
    """Join touching fill fragments while preserving every exact boundary path."""

    hatch_indexes = [
        index
        for index, entity in enumerate(entities)
        if entity.kind == "HATCH" and (entity.boundary_paths or entity.points)
    ]
    if len(hatch_indexes) < 2:
        return entities

    parents = list(range(len(hatch_indexes)))
    bounds = [
        _paths_bounds(_entity_boundary_paths(entities[index]))
        for index in hatch_indexes
    ]

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_position, first_index in enumerate(hatch_indexes):
        first = entities[first_index]
        for second_position in range(first_position):
            second = entities[hatch_indexes[second_position]]
            if _hatch_style_key(first) != _hatch_style_key(second):
                continue
            if not _bounds_touch(bounds[first_position], bounds[second_position], tolerance):
                continue
            if _paths_share_point(
                _entity_boundary_paths(first),
                _entity_boundary_paths(second),
                tolerance,
            ):
                union(first_position, second_position)

    members: dict[int, list[int]] = {}
    for position in range(len(hatch_indexes)):
        members.setdefault(find(position), []).append(position)

    merged_by_index: dict[int, CadEntity] = {}
    skipped_indexes: set[int] = set()
    for positions in members.values():
        entity_indexes = [hatch_indexes[position] for position in positions]
        first_index = min(entity_indexes)
        first = entities[first_index]
        paths = [
            path
            for entity_index in entity_indexes
            for path in _entity_boundary_paths(entities[entity_index])
        ]
        first.boundary_paths = paths
        first.points = paths[0]
        if len(entity_indexes) > 1:
            first.extra["merged_fill_parts"] = len(entity_indexes)
        merged_by_index[first_index] = first
        skipped_indexes.update(entity_indexes)
        skipped_indexes.discard(first_index)

    return [
        merged_by_index.get(index, entity)
        for index, entity in enumerate(entities)
        if index not in skipped_indexes
    ]


def _join_pdf_paths(
    entities: list[CadEntity],
    *,
    tolerance: float,
) -> list[CadEntity]:
    """Join endpoint-connected PDF segments into editable polyline trails."""

    passthrough: list[CadEntity] = []
    grouped: dict[tuple[Any, ...], list[tuple[CadEntity, list[tuple[float, float]]]]] = (
        defaultdict(list)
    )
    for entity in entities:
        if entity.kind == "LINE" and entity.start and entity.end:
            points = [entity.start, entity.end]
        elif entity.kind == "LWPOLYLINE" and entity.points and not entity.closed:
            points = entity.points
        else:
            passthrough.append(entity)
            continue
        grouped[_geometry_join_key(entity)].append((entity, points))

    joined: list[CadEntity] = []
    for paths in grouped.values():
        joined.extend(_minimum_polyline_trails(paths, tolerance))
    return [*passthrough, *joined]


def _minimum_polyline_trails(
    paths: list[tuple[CadEntity, list[tuple[float, float]]]],
    tolerance: float,
) -> list[CadEntity]:
    """Cover each undirected endpoint graph with the minimum Euler trails."""

    if not paths:
        return []

    def vertex(point: tuple[float, float]) -> tuple[int, int]:
        grid = max(tolerance, 1e-6)
        return (int(round(point[0] / grid)), int(round(point[1] / grid)))

    edge_vertices = [
        (vertex(points[0]), vertex(points[-1]))
        for _entity, points in paths
    ]
    adjacency: dict[tuple[int, int], list[int]] = defaultdict(list)
    for edge_index, (start, end) in enumerate(edge_vertices):
        adjacency[start].append(edge_index)
        adjacency[end].append(edge_index)

    remaining = set(range(len(paths)))
    components: list[set[int]] = []
    while remaining:
        seed = next(iter(remaining))
        component: set[int] = set()
        stack = [seed]
        while stack:
            edge_index = stack.pop()
            if edge_index in component:
                continue
            component.add(edge_index)
            remaining.discard(edge_index)
            start, end = edge_vertices[edge_index]
            stack.extend(
                adjacent
                for endpoint in (start, end)
                for adjacent in adjacency[endpoint]
                if adjacent not in component
            )
        components.append(component)

    results: list[CadEntity] = []
    for component in components:
        results.extend(
            _component_polyline_trails(
                paths,
                edge_vertices,
                component,
                tolerance,
            )
        )
    return results


def _component_polyline_trails(
    paths: list[tuple[CadEntity, list[tuple[float, float]]]],
    edge_vertices: list[tuple[tuple[int, int], tuple[int, int]]],
    component: set[int],
    tolerance: float,
) -> list[CadEntity]:
    component_adjacency: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    edge_data: dict[
        tuple[str, int],
        tuple[tuple[int, int], tuple[int, int]],
    ] = {}
    for edge_index in sorted(component):
        edge_id = ("real", edge_index)
        start, end = edge_vertices[edge_index]
        edge_data[edge_id] = (start, end)
        component_adjacency[start].append(edge_id)
        component_adjacency[end].append(edge_id)

    odd_vertices = sorted(
        endpoint
        for endpoint, incident in component_adjacency.items()
        if len(incident) % 2 == 1
    )
    virtual_ids: set[tuple[str, int]] = set()
    for pair_index in range(0, len(odd_vertices), 2):
        edge_id = ("virtual", pair_index // 2)
        start = odd_vertices[pair_index]
        end = odd_vertices[pair_index + 1]
        edge_data[edge_id] = (start, end)
        component_adjacency[start].append(edge_id)
        component_adjacency[end].append(edge_id)
        virtual_ids.add(edge_id)

    start_vertex = next(iter(component_adjacency))
    stack_vertices = [start_vertex]
    stack_edges: list[
        tuple[tuple[str, int], tuple[int, int], tuple[int, int]]
    ] = []
    circuit: list[tuple[tuple[str, int], tuple[int, int], tuple[int, int]]] = []
    used: set[tuple[str, int]] = set()
    local_adjacency = {
        endpoint: incident.copy()
        for endpoint, incident in component_adjacency.items()
    }
    while stack_vertices:
        current = stack_vertices[-1]
        incident = local_adjacency[current]
        while incident and incident[-1] in used:
            incident.pop()
        if incident:
            edge_id = incident.pop()
            if edge_id in used:
                continue
            used.add(edge_id)
            start, end = edge_data[edge_id]
            destination = end if current == start else start
            stack_vertices.append(destination)
            stack_edges.append((edge_id, current, destination))
        else:
            stack_vertices.pop()
            if stack_edges:
                circuit.append(stack_edges.pop())
    circuit.reverse()

    if virtual_ids:
        split_index = next(
            index
            for index, (edge_id, _start, _end) in enumerate(circuit)
            if edge_id in virtual_ids
        )
        circuit = circuit[split_index + 1 :] + circuit[: split_index + 1]

    raw_trails: list[
        list[tuple[tuple[str, int], tuple[int, int], tuple[int, int]]]
    ] = []
    current_trail: list[
        tuple[tuple[str, int], tuple[int, int], tuple[int, int]]
    ] = []
    for step in circuit:
        if step[0] in virtual_ids:
            if current_trail:
                raw_trails.append(current_trail)
                current_trail = []
        else:
            current_trail.append(step)
    if current_trail:
        raw_trails.append(current_trail)

    results: list[CadEntity] = []
    for trail in raw_trails:
        points: list[tuple[float, float]] = []
        source_entities: list[CadEntity] = []
        for edge_id, traversal_start, _traversal_end in trail:
            edge_index = edge_id[1]
            source_entity, source_points = paths[edge_index]
            edge_start, _edge_end = edge_vertices[edge_index]
            oriented = (
                source_points
                if traversal_start == edge_start
                else list(reversed(source_points))
            )
            if points and _points_close(points[-1], oriented[0], tolerance):
                points.extend(oriented[1:])
            else:
                points.extend(oriented)
            source_entities.append(source_entity)

        if len(points) < 2:
            continue
        closed = _points_close(points[0], points[-1], tolerance)
        if closed:
            points = _without_duplicate_endpoint(points)
        extra = source_entities[0].extra.copy()
        if len(source_entities) > 1:
            extra["joined_pdf_parts"] = len(source_entities)
        results.append(
            CadEntity(
                kind="LWPOLYLINE",
                layer="PDF_Geometry",
                points=points,
                closed=closed,
                confidence=min(entity.confidence for entity in source_entities),
                extra=extra,
            )
        )
    return results


def _geometry_join_key(entity: CadEntity) -> tuple[Any, ...]:
    return (
        entity.layer,
        entity.extra.get("aci_color"),
        entity.extra.get("true_color"),
        entity.extra.get("source_pdf_layer"),
        round(float(entity.extra.get("line_width_pixels", 0.0)), 3),
    )


def _entity_boundary_paths(entity: CadEntity) -> list[list[tuple[float, float]]]:
    return entity.boundary_paths or ([entity.points] if entity.points else [])


def _hatch_style_key(entity: CadEntity) -> tuple[Any, ...]:
    return (
        entity.extra.get("aci_color"),
        entity.extra.get("true_color"),
        entity.extra.get("source_pdf_layer"),
        entity.extra.get("source_is_white", False),
    )


def _paths_bounds(
    paths: list[list[tuple[float, float]]],
) -> tuple[float, float, float, float]:
    points = [point for path in paths for point in path]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bounds_touch(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    return not (
        first[2] + tolerance < second[0]
        or second[2] + tolerance < first[0]
        or first[3] + tolerance < second[1]
        or second[3] + tolerance < first[1]
    )


def _paths_share_point(
    first: list[list[tuple[float, float]]],
    second: list[list[tuple[float, float]]],
    tolerance: float,
) -> bool:
    return any(
        _points_close(first_point, second_point, tolerance)
        for first_path in first
        for first_point in first_path
        for second_path in second
        for second_point in second_path
    )


def _colour_tuple_to_int(value: Any) -> int | None:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    channels = [
        max(0, min(255, int(round(float(channel) * 255.0))))
        for channel in value[:3]
    ]
    return (channels[0] << 16) | (channels[1] << 8) | channels[2]


def _span_color(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

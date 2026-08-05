"""AI CAD Converter: raster/PDF drawings to editable CAD entities."""

from .settings import ConversionConfig

__all__ = ["CADConverter", "ConversionConfig", "DocumentResult", "PageResult"]


def __getattr__(name: str):
    """Avoid importing OpenCV-dependent modules until conversion is requested."""

    if name in {"CADConverter", "DocumentResult", "PageResult"}:
        from .orchestrator import CADConverter, DocumentResult, PageResult

        return {
            "CADConverter": CADConverter,
            "DocumentResult": DocumentResult,
            "PageResult": PageResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

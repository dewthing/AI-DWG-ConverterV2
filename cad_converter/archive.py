"""Small helpers for handing conversion results to a user."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def create_output_archive(output_directory: str | Path, archive_path: str | Path) -> Path:
    """Create a portable ZIP without recursively including itself."""

    source = Path(output_directory).resolve()
    target = Path(archive_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for file_path in sorted(source.rglob("*")):
            if not file_path.is_file() or file_path.resolve() == target:
                continue
            archive.write(file_path, file_path.relative_to(source))
    return target


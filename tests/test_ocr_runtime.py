from __future__ import annotations

import pytesseract

from cad_converter.ocr import _configure_tesseract


def test_configured_tesseract_executable_is_used(monkeypatch, tmp_path):
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"test executable placeholder")
    monkeypatch.setenv("AI_CAD_TESSERACT_CMD", str(executable))
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")

    resolved = _configure_tesseract()

    assert resolved == executable.resolve()
    assert pytesseract.pytesseract.tesseract_cmd == str(executable.resolve())

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cad_converter.image_io import iter_document


def test_decoded_page_safety_limit_rejects_oversized_input(tmp_path):
    source = tmp_path / "large.png"
    cv2.imwrite(str(source), np.full((100, 100, 3), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="safety limit"):
        list(iter_document(source, max_page_pixels=5_000))

import io

import numpy as np
import pytest
from scipy.io import savemat

from scripts.prepare_totaltext_development import audit, read_annotations


def row(name, bits):
    return {
        "id": name,
        "source_id": name,
        "image_sha256": name,
        "pixel_sha256": name,
        "dhash": bits,
    }


def test_audit_quarantines_external_overlap_and_groups_similar_sources():
    a, b, c, d = row("a", 0), row("b", 1), row("c", 2**64 - 1), row("d", 2**64 - 2)
    retained, report = audit([a, b, c, d], [row("external", 0)])
    assert report["excluded"] == 2
    assert {r["id"] for r in retained} == {"c", "d"}
    assert len({r["group_id"] for r in retained}) == 1
    assert len({r["split"] for r in retained}) == 1
    assert report["training_ready"] is False
    with pytest.raises(ValueError, match="Duplicate development ID"):
        audit([a, a], [])


def test_audit_detects_source_and_decoded_pixel_matches_despite_different_bytes():
    candidate = row("candidate", 0)
    external = row("external", 2**64 - 1)
    for key in ("source_id", "pixel_sha256", "image_sha256"):
        assert audit([candidate], [{**external, key: candidate[key]}])[1]["excluded"] == 1


def test_refined_mat_annotations_preserve_text_polygons_and_ignored_regions():
    words = np.empty((2, 6), dtype=object)
    for index, text in enumerate(("Hello", "#")):
        words[index] = [
            np.array(["x:"]),
            np.array([[1, 3, 2]]),
            np.array(["y:"]),
            np.array([[2, 2, 4]]),
            np.array([text]),
            np.array(["h" if index == 0 else "#"]),
        ]
    content = io.BytesIO()
    savemat(content, {"gt": words})
    result = read_annotations(content.getvalue())
    assert result[0]["text"] == "Hello"
    assert result[0]["polygon"] == [(1, 2), (3, 2), (2, 4)]
    assert not result[0]["ignore"] and result[1]["ignore"]

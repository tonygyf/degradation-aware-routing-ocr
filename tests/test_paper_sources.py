import pytest

from src.datasets.paper_sources import audit_sources, canonical_source


def test_canonical_parent_source_ignores_paths_case_and_derived_variant_ids():
    a = {
        "source_corpus": "ICDAR_2013",
        "source_image": "C:\\data\\IMG_1.JPG",
        "image_id": "arbitrary_variant",
    }
    b = {"source_corpus": "ic13", "source_image": "elsewhere/img_1.png"}
    assert canonical_source(a) == canonical_source(b) == "icdar2013:img_1"
    assert canonical_source({**b, "source_corpus": "sroie"}) != canonical_source(a)
    with pytest.raises(KeyError, match="source_image"):
        canonical_source({"source_corpus": "sroie", "image_id": "x_crop"})
    with pytest.raises(ValueError, match="filename"):
        canonical_source({"source_corpus": "sroie", "source_image": "a/"})


def test_audit_counts_unique_sources_and_filters_development_without_touching_test():
    def row(key, role, image):
        return {"id": key, "role": role, "source_corpus": "sroie", "source_image": image}

    baseline = [row("bt", "Test", "X.JPG"), row("bc", "Cal", "Y.png")]
    shc = [
        row("e1", "Exemplar", "x.png"),
        row("e2", "Exemplar", "X.JPG"),
        row("d", "Tau_dev", "y.jpg"),
        row("t", "Test", "Y.png"),
    ]
    r = audit_sources(baseline, shc)
    assert r["baseline_to_shc_unique_source_overlap"]["Test"]["Exemplar"] == 1
    assert r["shc_unique_parent_sources"] == 2
    assert r["shc_parents_in_multiple_roles"] == 1
    assert r["shc_development_sensitivity"]["Exemplar"]["remove_baseline_test_ids"] == []
    assert r["shc_development_sensitivity"]["Tau_dev"]["remove_baseline_test_ids"] == ["d"]
    assert r["shc_development_sensitivity"]["Tau_dev"]["also_remove_shc_test_ids"] == []
    assert r["baseline_development_without_shc_test"]["Cal"] == []
    assert r["frozen_test_ids"] == {"baseline": ["bt"], "shc": ["t"]}
    assert shc[-1]["role"] == "Test"
    with pytest.raises(ValueError, match="unique"):
        audit_sources(baseline, [shc[0], shc[0]])

from scripts.audit_sroie_source import summarize


def test_source_audit_detects_overlap_and_cross_split_identity_without_creating_roles():
    rows = [
        {
            "key": "same",
            "source_split": "train",
            "image_sha256": "a",
            "region_count": 2,
            "box_count": 2,
        },
        {
            "key": "same",
            "source_split": "test",
            "image_sha256": "a",
            "region_count": 1,
            "box_count": 0,
        },
        {
            "key": "X51006619503",
            "source_split": "train",
            "image_sha256": "b",
            "region_count": 0,
            "box_count": 0,
        },
    ]
    report = summarize(rows, {"a"})
    assert report["duplicate_id_rows"] == 1
    assert report["duplicate_image_byte_rows"] == 1
    assert report["id_cross_source_splits"] == 1
    assert report["image_byte_cross_source_splits"] == 1
    assert report["source_splits"]["test"]["region_box_mismatch"] == 1
    assert report["source_splits"]["train"]["ccocr_byte_overlap"] == 1
    assert report["source_splits"]["train"]["contains_paper_excluded_id"] == 1
    assert not report["author_role_manifest_available"]

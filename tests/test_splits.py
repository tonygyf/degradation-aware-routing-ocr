import pytest

from scripts.split_manifest import assign


def test_split_deterministic_order_independent_and_grouped():
    rows = [{"group_id": f"g{i // 2}", "image_sha256": str(i)} for i in range(40)]
    first = assign(rows)
    assert first == list(reversed(assign(list(reversed(rows)))))
    assert all(first[i]["split"] == first[i + 1]["split"] for i in range(0, 40, 2))
    with pytest.raises(ValueError, match="frozen"):
        assign(first)


def test_duplicate_content_requires_group_reconciliation():
    rows = [{"group_id": f"g{i}", "image_sha256": "same"} for i in range(40)]
    with pytest.raises(ValueError, match="Duplicate"):
        assign(rows)

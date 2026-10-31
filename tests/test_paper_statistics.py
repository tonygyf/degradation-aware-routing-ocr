import copy

import pytest

from src.evaluation.bootstrap import cluster_flag_interval, paired_sign_flip, paired_stratified
from src.evaluation.ccocr_diagnostics import diagnostics, repetition_flag


def test_paired_strata_preserve_sizes_and_difference_columns():
    values = [[1, 0], [1, 0], [-1, 0]]
    result = paired_stratified(values, ["large", "large", "small"], resamples=100)
    assert result["estimate"] == [1 / 3, 0]
    assert result["ci95_low"] == result["estimate"] == result["ci95_high"]
    assert result == paired_stratified(values, ["large", "large", "small"], resamples=100)
    with pytest.raises(ValueError, match="aligned"):
        paired_stratified(values, ["wrong"])
    with pytest.raises(ValueError, match="finite"):
        paired_stratified([[float("nan")]], ["a"])


def test_cluster_resampling_retains_all_descendants_and_variable_denominator():
    flags, clusters = [0] * 9 + [1], ["large"] * 9 + ["small"]
    result = cluster_flag_interval(flags, clusters, resamples=1000)
    assert result["estimate"] == 0.1
    assert result["ci95_low"] == 0
    assert result["ci95_high"] == 1
    assert result["clusters"] == 2
    singleton = cluster_flag_interval(flags, ["same"] * 10, resamples=50)
    assert singleton["ci95_low"] == singleton["ci95_high"] == 0.1


def test_repetition_requires_three_adjacent_blocks_in_allowed_range():
    assert repetition_flag("prefix A B A\nB A B suffix")
    assert not repetition_flag("A B A B")
    assert not repetition_flag("A B gap A B gap2 A B")
    assert not repetition_flag("A A A")
    assert repetition_flag("A A A A A A")
    assert not repetition_flag("A B a b A B")
    block = " ".join(str(i) for i in range(20))
    assert repetition_flag(" ".join([block] * 3))
    block = " ".join(str(i) for i in range(21))
    assert not repetition_flag(" ".join([block] * 3))


def test_diagnostics_align_by_id_and_reject_unpaired_observations():
    records = [
        {
            "id": str(i),
            "system": system,
            "reference": "A",
            "prediction": "A",
            "subset": "one" if i < 2 else "two",
            "latency_ms": 1,
            "vlm_called": system != "paddleocr",
        }
        for system in ("qwen", "confroute", "paddleocr")
        for i in range(3)
    ]
    result = diagnostics(list(reversed(records)), resamples=100)
    assert result["comparisons"]["confroute_minus_qwen"]["eval_trans"]["ci95_high"] == 0
    assert result["comparisons"]["confroute_minus_paddleocr"]["vlm_call_rate"]["estimate"] == 1
    assert result["repetition"]["qwen"]["count"] == 0
    with pytest.raises(ValueError, match="identical"):
        diagnostics(records[:-1])
    mismatch = copy.deepcopy(records)
    mismatch[0]["reference"] = "different"
    with pytest.raises(ValueError, match="references"):
        diagnostics(mismatch)


def test_sign_flip_handles_ties_and_approximates_exact_small_sample():
    assert paired_sign_flip([0, 0], permutations=100)["p_two_sided"] == 1
    result = paired_sign_flip([1, 1, 1], permutations=10000, seed=42)
    assert result["p_two_sided"] == pytest.approx(0.25, abs=0.02)
    assert result == paired_sign_flip([1, 1, 1], permutations=10000, seed=42)
    with pytest.raises(ValueError, match="finite"):
        paired_sign_flip([float("inf")])
    with pytest.raises(ValueError, match="positive"):
        paired_sign_flip([1], permutations=0)

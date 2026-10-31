import pytest

from src.recognition.internvl_preprocess import choose_grid


def test_grid_respects_orientation_budget_and_large_image_tie_break():
    assert choose_grid(448, 448) == (1, 1)
    assert choose_grid(1000, 1000) == (3, 3)
    assert choose_grid(4480, 448) == (10, 1)
    assert choose_grid(448, 4480) == (1, 10)
    assert choose_grid(10000, 1)[0] * choose_grid(10000, 1)[1] <= 12
    assert choose_grid(1000, 1000, max_tiles=1) == (1, 1)
    with pytest.raises(ValueError, match="positive integers"):
        choose_grid(0, 1)

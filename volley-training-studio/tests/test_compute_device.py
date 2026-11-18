import pytest

from app.pipeline import _select_compute_device


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(True, 0), (False, "cpu")],
)
def test_select_compute_device(cuda_available, expected):
    assert _select_compute_device(cuda_available) == expected

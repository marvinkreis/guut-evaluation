from kth import kth

def test__kth():
    assert kth([3, 1, 2, 5, 4], 4) == 5, "Should return the maximum: 5"
    assert kth([6, 3, 5, 7, 2], 2) == 5, "Should return the third-lowest: 5"
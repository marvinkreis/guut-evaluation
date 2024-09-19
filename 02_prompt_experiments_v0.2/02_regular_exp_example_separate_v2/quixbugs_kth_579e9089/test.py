from kth import kth

def test__kth():
    """The mutant should not return the correct kth-lowest element due to the logic change."""
    # Expected correct outputs
    assert kth([3, 1, 2, 5, 4], 1) == 2, "For k=1, expected 2"
    assert kth([3, 1, 2, 5, 4], 2) == 3, "For k=2, expected 3"
    assert kth([3, 1, 2, 5, 4], 3) == 4, "For k=3, expected 4"
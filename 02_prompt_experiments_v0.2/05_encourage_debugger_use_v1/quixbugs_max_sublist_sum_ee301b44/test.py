from max_sublist_sum import max_sublist_sum

def test__max_sublist_sum():
    """The mutant does not reset max_ending_here correctly, leading to incorrect sums."""
    result = max_sublist_sum([4, -5, 2, 1, -1, 3])
    expected = 5
    assert result == expected, f"expected {expected}, got {result}"
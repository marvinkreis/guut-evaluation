from max_sublist_sum import max_sublist_sum

def test__max_sublist_sum():
    """The mutant's failure to reset 'max_ending_here' leads to incorrect outputs."""
    # Test with a typical input that includes both negative and positive integers.
    output = max_sublist_sum([4, -5, 2, 1, -1, 3])
    assert output == 5, "Expected maximum sublist sum is 5."

    # Test with another input that yields a maximum sublist sum greater than 0.
    output = max_sublist_sum([-1, -2, -3, -4, -5, 10, -1, 1, 2])
    assert output == 12, "Expected maximum sublist sum is 12."
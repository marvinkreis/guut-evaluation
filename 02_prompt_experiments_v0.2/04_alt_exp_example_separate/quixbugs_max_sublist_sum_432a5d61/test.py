from max_sublist_sum import max_sublist_sum

def test__max_sublist_sum():
    """The mutant fails for the input [4, -5, 2, 1, -1, 3] due to incorrect handling of negative sums."""
    output = max_sublist_sum([4, -5, 2, 1, -1, 3])
    assert output == 5, "max_sublist_sum should correctly compute the maximum sublist sum"
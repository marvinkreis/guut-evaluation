from max_sublist_sum import max_sublist_sum

def test__max_sublist_sum():
    """The mutant incorrectly calculates the maximum sublist sum by not resetting 'max_ending_here' when it goes negative."""
    output = max_sublist_sum([4, -5, 2, 1, -1, 3])
    assert output == 5, "The maximum sublist sum must be 5."
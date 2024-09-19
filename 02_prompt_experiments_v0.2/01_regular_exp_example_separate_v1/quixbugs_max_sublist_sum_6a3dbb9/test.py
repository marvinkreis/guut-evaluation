from max_sublist_sum import max_sublist_sum

def test__max_sublist_sum():
    """The mutant fails to reset the accumulating sum correctly, resulting in an incorrect max sublist sum."""
    output = max_sublist_sum([-1, -2, -3, 4, 5])
    assert output > 8, "Expected a max sublist sum greater than 8, indicating that the mutant is faulty."
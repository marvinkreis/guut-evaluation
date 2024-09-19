from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted():
    """The mutant changes the while condition, leading to an IndexError for empty arrays."""
    # Test case with an empty array
    assert find_first_in_sorted([], 1) == -1, "The result should be -1 when searching in an empty array."
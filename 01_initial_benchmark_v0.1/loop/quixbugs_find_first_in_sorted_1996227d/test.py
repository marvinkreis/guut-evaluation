from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted():
    assert find_first_in_sorted([], 1) == -1, "Expected: -1 for empty list"
    assert find_first_in_sorted([3, 4, 5], 10) == -1, "Expected: -1 for value larger than any in the list"
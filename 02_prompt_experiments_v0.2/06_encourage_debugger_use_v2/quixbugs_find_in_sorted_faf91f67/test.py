from find_in_sorted import find_in_sorted

def test__find_in_sorted():
    """Changing 'binsearch(mid + 1, end)' to 'binsearch(mid, end)' would cause infinite recursion when searching for a value not present in the array."""
    output = find_in_sorted([1, 2, 3, 4, 5, 6, 6, 7], 8)
    assert output == -1, "Expected -1 since 8 is not in the array"
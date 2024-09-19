from find_in_sorted import find_in_sorted

def test__find_in_sorted():
    """Mutant changes binary search and causes infinite recursion for values not found."""
    found_index = find_in_sorted([3, 4, 5, 5, 5, 5, 6], 5)
    not_found_index = find_in_sorted([3, 4, 5, 5, 5, 5, 6], 7)
    
    assert found_index == 3, "Should return the index of 5"
    assert not_found_index == -1, "Should return -1 for value not found"
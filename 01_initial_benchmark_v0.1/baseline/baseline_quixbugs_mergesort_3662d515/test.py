from mergesort import mergesort

def test_mergesort():
    # Test with an empty list, which should return an empty list
    result = mergesort([])
    expected = []
    assert result == expected, f"Expected {expected}, but got {result}"
    
    # Test with a non-empty list to ensure correct behavior
    result = mergesort([3, 1, 2])
    expected = [1, 2, 3]
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test with a single element list
    result = mergesort([5])
    expected = [5]
    assert result == expected, f"Expected {expected}, but got {result}"
    
    # Test with a larger list
    result = mergesort([4, 3, 2, 1])
    expected = [1, 2, 3, 4]
    assert result == expected, f"Expected {expected}, but got {result}"
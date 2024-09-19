from quicksort import quicksort

def test__quicksort():
    # Test input with duplicate values
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    expected = sorted(arr)  # using Python's built-in sorted for comparison
    result = quicksort(arr)
    
    assert result == expected, f"Expected {expected}, but got {result}"
    
    # Test input with negative numbers
    arr_neg = [-3, -1, -4, -1, -5, -9, -2, -6, -5, -3, -5]
    expected_neg = sorted(arr_neg)
    result_neg = quicksort(arr_neg)
    
    assert result_neg == expected_neg, f"Expected {expected_neg}, but got {result_neg}"

    # Test input with already sorted numbers
    arr_sorted = [1, 2, 3, 4, 5]
    expected_sorted = sorted(arr_sorted)
    result_sorted = quicksort(arr_sorted)
    
    assert result_sorted == expected_sorted, f"Expected {expected_sorted}, but got {result_sorted}"

    # Test input with a single value
    arr_single = [42]
    expected_single = sorted(arr_single)
    result_single = quicksort(arr_single)
    
    assert result_single == expected_single, f"Expected {expected_single}, but got {result_single}"

    # Test empty input
    arr_empty = []
    expected_empty = sorted(arr_empty)
    result_empty = quicksort(arr_empty)
    
    assert result_empty == expected_empty, f"Expected {expected_empty}, but got {result_empty}"
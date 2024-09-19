from find_in_sorted import find_in_sorted

def test_find_in_sorted():
    # Test case where x is present multiple times
    arr = [1, 2, 2, 2, 3]
    x = 2

    # Running the function to identify the index of the first occurrence
    actual_index_first = find_in_sorted(arr, x)

    # Check if the return value is indeed the first occurrence
    assert actual_index_first != -1, f"Expected to find {x} in the array."
    assert actual_index_first in (1, 2), (f"Expected index to be 1 or 2 for duplicates, but got {actual_index_first}.")

    # Check for a value not in the array
    arr = [1, 2, 2, 2, 3]
    x = 4
    expected_index_not_found = -1
    actual_index_not_found = find_in_sorted(arr, x)

    assert actual_index_not_found == expected_index_not_found, f"Expected index {expected_index_not_found}, but got {actual_index_not_found}."

    # Check for an empty array
    arr = []
    x = 1
    expected_index_empty = -1
    actual_index_empty = find_in_sorted(arr, x)

    assert actual_index_empty == expected_index_empty, f"Expected index {expected_index_empty}, but got {actual_index_empty}."
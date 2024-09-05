from bucketsort import bucketsort

def test_bucketsort():
    # Test with a simple input
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    k = 10  # Elements are in the range [0, 10)
    expected_output = sorted(arr)  # The expected sorted output is the sorted version of arr

    # Execute the bucketsort function
    result = bucketsort(arr, k)

    # Assert the result is as expected
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Additional test to catch the mutant
    arr_with_duplicate = [2, 2, 2, 1, 1, 0]
    k_duplicate = 3  # The values in arr are in the range [0, 3)
    expected_output_duplicate = sorted(arr_with_duplicate)  # Expected: [0, 1, 1, 2, 2, 2]

    # Execute the bucketsort function
    result_duplicate = bucketsort(arr_with_duplicate, k_duplicate)

    # Assert the result is as expected
    assert result_duplicate == expected_output_duplicate, f"Expected {expected_output_duplicate}, but got {result_duplicate}"
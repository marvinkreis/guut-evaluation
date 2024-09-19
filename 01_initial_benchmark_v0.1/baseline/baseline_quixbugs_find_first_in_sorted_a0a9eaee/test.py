from find_first_in_sorted import find_first_in_sorted

def test_find_first_in_sorted():
    # Test case to find the first occurrence of 5
    arr = [3, 4, 5, 5, 5, 5, 6]
    x = 5
    expected_index = 2
    actual_index = find_first_in_sorted(arr, x)
    assert actual_index == expected_index, f"Expected index {expected_index} but got {actual_index}"

    # Test case where x is not present
    arr = [1, 2, 3]
    x = 4
    expected_index = -1
    actual_index = find_first_in_sorted(arr, x)
    assert actual_index == expected_index, f"Expected index {expected_index} but got {actual_index}"

    # Edge case where the array is empty
    arr = []
    x = 1
    expected_index = -1
    actual_index = find_first_in_sorted(arr, x)
    assert actual_index == expected_index, f"Expected index {expected_index} but got {actual_index}"

    # Edge case where the array has one element which is the target
    arr = [1]
    x = 1
    expected_index = 0
    actual_index = find_first_in_sorted(arr, x)
    assert actual_index == expected_index, f"Expected index {expected_index} but got {actual_index}"

    # Edge case where the array has one element which is not the target
    arr = [2]
    x = 1
    expected_index = -1
    actual_index = find_first_in_sorted(arr, x)
    assert actual_index == expected_index, f"Expected index {expected_index} but got {actual_index}"

    # Additional edge case with consecutive duplicate elements
    arr = [5, 5, 5, 5, 5]
    x = 5
    expected_index = 0
    actual_index = find_first_in_sorted(arr, x)
    assert actual_index == expected_index, f"Expected index {expected_index} but got {actual_index}"

    # Specific case to check boundaries with the mutant
    # Test with a custom array to expose the mutant behavior
    arr = [1, 2, 3, 5, 5, 5, 6]  # Sorted array with occurrences of 5
    x = 5
    expected_index = 3  # The first index of 5 is 3
    actual_index = find_first_in_sorted(arr, x)
    
    # Ensure the correct behavior
    assert actual_index == expected_index, f"Expected index {expected_index} but got {actual_index}"
    
    # Now we will check with the mutant condition
    # Since the mutant introduces an index error, we can catch it.
    try:
        # This should not raise any error for the correct implementation
        find_first_in_sorted(arr, 2)
    except IndexError:
        assert False, "Mutant caused an IndexError, which is not expected in the correct implementation."

# Call the test function to run the assertions
test_find_first_in_sorted()
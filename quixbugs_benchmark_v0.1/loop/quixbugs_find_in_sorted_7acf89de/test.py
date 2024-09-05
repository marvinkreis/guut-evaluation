from find_in_sorted import find_in_sorted

def test__find_in_sorted_mutant_detection():
    # This test should pass with the correct implementation and fail with the mutant
    test_array = [1, 2, 3, 4, 5]
    test_value = 6  # Value that doesn't exist in the array
    assert find_in_sorted(test_array, test_value) == -1, "Output should be -1 for non-existent values"
from max_sublist_sum import max_sublist_sum

def test_max_sublist_sum():
    # This is a specific case where handling of negative values should matter
    test_array = [4, -5, 2, 1, -1, 3]
    expected_result = 5  # The maximum sum of the subarray [2, 1, -1, 3]
    
    result = max_sublist_sum(test_array)
    
    # Assert that the function returns the expected maximum sublist sum
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
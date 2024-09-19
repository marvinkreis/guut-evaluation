from lis import lis

def test_lis():
    # Test case to detect the mutant
    input_array = [4, 1, 5, 3, 7, 6, 2]
    
    # The expected output is 3, since the longest increasing subsequence is [1, 5, 7]
    expected_output = 3
    assert lis(input_array) == expected_output

    # Edge case: The longest increasing subsequence is at the beginning
    input_array_2 = [1, 2, 3, 4, 5]
    expected_output_2 = 5  # Whole array is increasing
    assert lis(input_array_2) == expected_output_2
    
    # Edge case: The longest increasing subsequence is the last three elements
    input_array_3 = [5, 3, 4, 2, 6]
    expected_output_3 = 3  # Longest increasing subsequence is [4, 5, 6]
    assert lis(input_array_3) == expected_output_3
    
    # Edge case: An array with no increasing subsequence
    input_array_4 = [5, 4, 3, 2, 1]
    expected_output_4 = 1  # Only one element at a time can be taken
    assert lis(input_array_4) == expected_output_4
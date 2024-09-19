from kheapsort import kheapsort

def test_kheapsort():
    # Create a test case where elements that are at most k places from sorted positions
    arr = [3, 2, 1, 5, 4]  # Valid input for k-heapsort with k=2
    k = 2
    
    # Expected output is a fully sorted list
    expected_output = [1, 2, 3, 4, 5]  
    
    # Collect the sorted result from kheapsort
    sorted_output = list(kheapsort(arr, k))
    
    # Assert that the output is as expected
    assert sorted_output == expected_output, f"Expected {expected_output}, but got {sorted_output}"

    # Test case with larger k to further differentiate behavior
    arr2 = [1, 5, 3, 2, 4]  # Still valid input for k-heapsort, should sort correctly
    k2 = 3
    expected_output2 = [1, 2, 3, 4, 5]
    
    sorted_output2 = list(kheapsort(arr2, k2))
    
    # Make assertions
    assert sorted_output2 == expected_output2, f"Expected {expected_output2}, but got {sorted_output2}"

    # Test an already sorted array with k = 0 which should return the same array
    arr3 = [1, 2, 3, 4, 5]
    k3 = 0
    expected_output3 = [1, 2, 3, 4, 5]
    
    sorted_output3 = list(kheapsort(arr3, k3))
    
    # Assert sorted output
    assert sorted_output3 == expected_output3, f"Expected {expected_output3}, but got {sorted_output3}"

    # Edge case with all identical removals to highlight mutant
    arr4 = [1, 2, 3, 4, 5, 6]  # Out-of-order elements with k=1
    k4 = 1
    expected_output4 = [1, 2, 3, 4, 5, 6]  # Already sorted
    
    sorted_output4 = list(kheapsort(arr4, k4))

    assert sorted_output4 == expected_output4, f"Expected {expected_output4}, but got {sorted_output4}"
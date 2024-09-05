from subsequences import subsequences

def test_subsequences():
    # Test when k is 0
    result = subsequences(a=1, b=5, k=0)
    expected_result = [[]]
    assert result == expected_result, f"Expected {expected_result} but got {result}"

    # Additional test for k > 0 to ensure proper functioning
    result = subsequences(a=1, b=5, k=3)
    expected_result = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    assert result == expected_result, f"Expected {expected_result} but got {result}"
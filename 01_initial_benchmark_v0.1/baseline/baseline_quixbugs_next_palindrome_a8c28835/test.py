from next_palindrome import next_palindrome

def test_next_palindrome():
    # Test case that detects the mutant
    input_digits = [9, 9, 9]  # Current palindrome
    expected_output = [1, 0, 0, 1]  # The next palindrome
    result = next_palindrome(input_digits)
    
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Additional check for a non-boundary case
    input_digits2 = [1, 2, 1]  # Another current palindrome
    expected_output2 = [1, 3, 1]  # The next palindrome
    result2 = next_palindrome(input_digits2)
    
    assert result2 == expected_output2, f"Expected {expected_output2}, but got {result2}"
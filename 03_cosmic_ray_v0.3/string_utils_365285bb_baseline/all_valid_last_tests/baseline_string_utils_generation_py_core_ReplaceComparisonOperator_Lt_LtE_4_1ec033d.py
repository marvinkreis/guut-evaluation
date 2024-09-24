from string_utils.generation import roman_range

def test__roman_range_mutant_detection():
    # This test is designed to pass with the correct code and fail with the mutant.

    # Test case 1: start equals stop with negative step
    try:
        # This should raise an OverflowError in the original code
        result = list(roman_range(start=5, stop=5, step=-1))  
        assert False, "Expected OverflowError was not raised in original code"
    except OverflowError:
        pass  # This means the original code correctly raises OverflowError

    # Test case 2: start < stop with a negative step
    try:
        # This should also raise OverflowError in the original code
        result = list(roman_range(start=1, stop=5, step=-1))  
        assert False, "Expected OverflowError was not raised in original code"
    except OverflowError:
        pass  # This should pass in the original code

    # Test case 3: valid range (should pass in both versions)
    try:
        result = list(roman_range(start=1, stop=5, step=1))
        expected_result = ['I', 'II', 'III', 'IV', 'V']  # Expected output based on roman_encode
        assert result == expected_result, f"Expected {expected_result}, got {result}"
    except Exception as e:
        assert False, f"Unexpected exception raised: {type(e).__name__}: {str(e)}"

# To run the test function directly
if __name__ == "__main__":
    test__roman_range_mutant_detection()
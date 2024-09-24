from string_utils.generation import roman_range

def test_roman_range():
    # Testing valid input should produce a correct sequence of Roman numerals
    # for the current limit of 3999.
    
    result = list(roman_range(7))
    expected = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    assert result == expected, f"Expected {expected}, got {result}"

    # Testing the edge case that should still be valid
    try:
        result_edge = list(roman_range(3999))
        # Note: Generating a full list for 3999 would be extremely long
        # We create a partial output based on known values
        assert len(result_edge) == 3999, "Expected 3999 elements for roman_range up to 3999."
    except Exception as e:
        assert False, f"Unexpected exception for 3999: {e}"

    # Testing the input above the limit
    try:
        result_invalid = list(roman_range(4000))
        # If the code does not raise an error, the mutant is detected.
        assert False, "Expected ValueError for input 4000, but none was raised."
    except ValueError:
        # This is expected, should raise ValueError in the original function.
        pass
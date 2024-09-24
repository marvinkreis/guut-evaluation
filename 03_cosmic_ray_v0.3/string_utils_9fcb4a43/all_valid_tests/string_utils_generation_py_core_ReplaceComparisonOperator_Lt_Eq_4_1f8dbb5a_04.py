from string_utils.generation import roman_range

def test__roman_range_backward_exceed():
    """Test for a case where start equals stop with a negative step."""
    try:
        result = list(roman_range(start=3, stop=3, step=-1))
        assert False, "Expected OverflowError for equal start and stop with negative step."
    except OverflowError as e:
        assert str(e) == 'Invalid start/stop/step configuration', f"Unexpected exception message: {str(e)}"

def _test__roman_range_with_negative_step():
    """Test backward moving with step adjustment from a valid configuration."""
    # Valid backward range demonstration (valid case should output Roman numerals)
    expected_output = ['V', 'IV', 'III', 'II', 'I']
    result = list(roman_range(start=5, stop=1, step=-1))
    assert result == expected_output, f"Expected {expected_output}, got {result}."

def _test__roman_range_edge_case_forward():
    """Test an invalid configuration for forward movement (edge cases for equality)."""
    try:
        result = list(roman_range(start=1, stop=1, step=1))
        # This should yield simply ['I']
        assert result == ['I'], f"Expected ['I'], got {result}."
    except Exception as e:
        assert False, f"Unexpected exception raised: {str(e)}"

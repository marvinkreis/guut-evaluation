from string_utils.generation import roman_range

def test__roman_range_invalid_step():
    """Test an invalid backward range with equal start and stop values to trigger an exception."""
    try:
        output = list(roman_range(start=5, stop=5, step=-1))
        assert False, "Expected OverflowError for equal start and stop with backward step."
    except OverflowError as e:
        assert str(e) == 'Invalid start/stop/step configuration', f"Unexpected exception message: {str(e)}"

def test__roman_range_exceeds():
    """Test a backward range where the start exceeds the stop."""
    try:
        output = list(roman_range(start=5, stop=1, step=-1))
        assert output == ['V', 'IV', 'III', 'II', 'I'], "Output does not match expected Roman numerals."
    except Exception as e:
        assert False, f"Unexpected exception raised: {str(e)}"
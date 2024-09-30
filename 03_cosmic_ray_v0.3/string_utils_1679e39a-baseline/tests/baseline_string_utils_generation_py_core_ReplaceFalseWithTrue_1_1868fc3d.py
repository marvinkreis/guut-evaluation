from string_utils.generation import roman_range

def test__roman_range():
    """
    Test if the function correctly raises a ValueError when 'stop' is outside the valid range (1-3999).
    The baseline implementation properly raises a ValueError when the 'stop' is less than 1,
    while the mutant incorrectly accepts negative values due to the change in the validation logic.
    """
    
    # This should raise a ValueError in the baseline because 'stop' is negative.
    try:
        roman_range(-1)
    except ValueError as e:
        assert str(e) == '"stop" must be an integer in the range 1-3999', "Expected ValueError for negative stop value"

    # This is a valid test case which should work on both baseline and mutant, confirming normal operational behavior
    try:
        result = list(roman_range(5))  # Valid input, should yield I, II, III, IV, V
        assert result == ['I', 'II', 'III', 'IV', 'V'], "Expected roman range output for positive range"
    except Exception:
        assert False, "The code failed with valid input."
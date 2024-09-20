from string_utils.generation import roman_range

def test__roman_range_invalid_input():
    """The mutant incorrectly allows values above 3999 which should not create roman numerals."""
    try:
        roman_range(4000)
    except ValueError as e:
        # Verify the exception message from the correct implementation.
        assert str(e) == '"stop" must be an integer in the range 1-3999'
    else:
        assert False, "Expected ValueError was not raised for input 4000"
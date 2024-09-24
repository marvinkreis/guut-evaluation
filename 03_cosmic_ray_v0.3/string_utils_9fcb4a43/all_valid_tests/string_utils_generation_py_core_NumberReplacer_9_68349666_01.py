from string_utils.generation import roman_range

def test__roman_range():
    """The mutant allows zero as a valid argument which should raise ValueError."""
    try:
        roman_range(0)
        assert False, "Expected ValueError not raised"
    except ValueError:
        pass  # This is expected
    
    try:
        roman_range(5, 1, 1)  # Should work fine
        assert True  # Valid range check to ensure success
    except Exception:
        assert False, "Unexpected error for valid input"
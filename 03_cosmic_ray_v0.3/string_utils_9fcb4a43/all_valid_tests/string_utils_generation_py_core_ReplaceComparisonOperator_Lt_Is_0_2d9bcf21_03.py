from string_utils.generation import roman_range

def test__roman_range():
    """The mutant allows zero-length backward ranges to succeed, whereas the correct code raises an error."""
    try:
        list(roman_range(5, 5, -1))
        assert False, "Expected OverflowError not raised."  # This is the expected behavior.
    except OverflowError:
        pass  # This is the expected behavior.
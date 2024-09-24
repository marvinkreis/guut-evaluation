from string_utils.generation import roman_range

def test__roman_range():
    """The mutant might allow invalid range configuration to produce results, while the correct code raises an error."""
    try:
        output = list(roman_range(10, 1, -1))
        assert False, "Expected OverflowError not raised."
    except OverflowError:
        pass  # This is the expected behavior, so we simply pass.
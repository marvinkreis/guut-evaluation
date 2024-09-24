from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant's change allows input of 4000 to be processed, unlike the correct code which raises a ValueError."""
    try:
        roman_encode(4000)
    except ValueError:
        return  # This should happen with the correct implementation
    assert False, "Expected ValueError not raised for input 4000, which should be invalid for Roman numerals."
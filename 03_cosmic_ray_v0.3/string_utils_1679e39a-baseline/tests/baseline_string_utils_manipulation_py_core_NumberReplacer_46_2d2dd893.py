from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the output of the roman_encode function with the input 4000, which should raise a ValueError since 
    the mutant code incorrectly accepts values up to 4000, while the original code only allows values up to 
    3999. This will ensure that we can detect the mutant.
    """
    try:
        roman_encode(4000)
        assert False, "Expected ValueError not raised"
    except ValueError:
        pass  # This is expected
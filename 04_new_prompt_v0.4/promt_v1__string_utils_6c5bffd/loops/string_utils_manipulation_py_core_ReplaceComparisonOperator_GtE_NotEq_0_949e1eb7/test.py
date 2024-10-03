from string_utils.manipulation import roman_decode

def test__roman_decode_mutant_killer():
    """
    Test decoding of valid Roman numeral strings with consecutive characters.
    The inputs 'III' and 'XX' are expected to return 3 and 20 respectively.
    The mutant will fail on these cases due to incorrect handling of equal sign values.
    """
    # Test cases where the baseline and mutant should behave differently
    assert roman_decode('III') == 3
    assert roman_decode('XX') == 20
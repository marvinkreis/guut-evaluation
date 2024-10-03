from string_utils.manipulation import roman_encode

def test__roman_encode_kill_mutant():
    """
    Test the encoding of the number 100, which should return 'C' in valid encoding.
    The mutant is expected to raise a KeyError due to a mapping error.
    """
    # This should succeed in the baseline
    output = roman_encode(100)
    assert output == 'C'
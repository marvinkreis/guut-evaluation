from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether the function correctly encodes the number 3999 into its Roman numeral equivalent 'MMMCMXCIX'.
    The mutant allows 3999 which should still be valid, but will fail to encode properly as specified in the mutant conditions,
    while the baseline should return 'MMMCMXCIX'. Using a valid input allows us to compare the output of both versions.
    """
    output = roman_encode(3999)
    assert output == 'MMMCMXCIX'
from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test encoding of the number 3999 into Roman numeral. The baseline should correctly encode it as 'MMMCMXCIX',
    while the mutant should raise a ValueError due to the input exceeding the new limit of 3998.
    """
    # This should pass on the baseline, encoding 3999 as MMMCMXCIX
    output = roman_encode(3999)
    assert output == 'MMMCMXCIX'
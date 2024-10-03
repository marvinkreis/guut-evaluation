from string_utils.manipulation import roman_encode

def test__roman_encode_tens():
    """
    Test encoding of 40 to see if the modification leads to incorrect Roman numeral representation.
    The input 40 should return 'XL' in the baseline, but will cause a KeyError in the mutant
    due to an invalid mapping for tens in Roman numeral encoding.
    """
    output = roman_encode(40)
    assert output == 'XL', f"Expected 'XL', but got {output}"
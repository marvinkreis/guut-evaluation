from string_utils.manipulation import roman_decode

def test__roman_decode():
    """
    Test the decoding of a Roman numeral string that includes values that will test both addition and subtraction:
    This test uses the Roman numeral 'XIV', which should return 14.
    The original logic processes the numeral correctly, while the mutant's implementation, which uses 'is' 
    instead of '>=' may result in incorrect behavior for such compound numerals.
    """
    output = roman_decode('XIV')
    assert output == 14
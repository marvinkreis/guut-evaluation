from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function to ensure it correctly encodes the number 5. 
    The mutant version alters the internal mappings, causing an error when encoding 
    5 due to the change in the mapping from {1: 'V', 5: 'V'} to {1: 'I', 6: 'V'}.
    The expected output for the baseline is 'V', while the mutant should lead to a KeyError.
    """
    output = roman_encode(5)
    assert output == 'V', "Expected output for 'roman_encode(5)' to be 'V'"
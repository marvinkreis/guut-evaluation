from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function with input 400.
    The baseline should return 'CD', while the mutant raises a KeyError due to the incorrect mapping for hundreds.
    """
    output = roman_encode(400)
    assert output == 'CD'
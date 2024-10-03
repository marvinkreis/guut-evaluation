from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with an input of 12. The baseline should return 'XII', while the mutant
    will raise a KeyError due to the modified mapping of the tens in the mutant version.
    """
    output = roman_encode(12)
    assert output == 'XII', f"Expected 'XII', got {output}"
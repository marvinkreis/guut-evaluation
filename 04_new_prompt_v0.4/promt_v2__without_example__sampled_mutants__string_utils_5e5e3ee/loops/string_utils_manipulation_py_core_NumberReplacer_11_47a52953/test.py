from string_utils.manipulation import roman_encode

def test__roman_encode_mutant_killer():
    """
    Test that the roman_encode function correctly encodes the number 400 as 'CD'.
    The mutant will fail this test due to an invalid mapping for the hundreds place.
    """
    output = roman_encode(400)
    assert output == 'CD', f"Expected 'CD', got {output}"
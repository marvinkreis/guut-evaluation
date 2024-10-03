from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function to verify that the encoding of the number 1 returns 'I'.
    The mutant's change in the mappings will lead to a KeyError when trying to encode 1.
    This test checks for correct behavior on valid input.
    """
    output = roman_encode(1)
    assert output == 'I', f"Expected 'I', got {output}"
from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant will fail when encoding the number 5 due to a key error in the mappings."""
    try:
        output = roman_encode(5)
        assert output == 'V', "Encoding error: should return 'V' for input 5"
    except KeyError as e:
        assert True  # The mutant should fail here due to the key error
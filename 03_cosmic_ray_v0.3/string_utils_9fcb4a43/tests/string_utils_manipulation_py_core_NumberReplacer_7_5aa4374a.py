from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant incorrectly encodes numbers with 4 in them due to altered mappings."""
    
    output_14 = roman_encode(14)
    assert output_14 == "XIV", f"Expected 'XIV', but got '{output_14}'"

    output_24 = roman_encode(24)
    assert output_24 == "XXIV", f"Expected 'XXIV', but got '{output_24}'"

    output_40 = roman_encode(40)
    assert output_40 == "XL", f"Expected 'XL', but got '{output_40}'"

    output_44 = roman_encode(44)
    assert output_44 == "XLIV", f"Expected 'XLIV', but got '{output_44}'"
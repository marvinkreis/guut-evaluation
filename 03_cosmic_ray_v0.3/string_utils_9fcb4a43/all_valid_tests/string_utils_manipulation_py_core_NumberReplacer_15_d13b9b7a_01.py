from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant should not handle the zero input correctly."""
    try:
        output = roman_encode(0)
        assert False, "roman_encode should raise an error for zero input."
    except ValueError as e:
        assert str(e) == 'Input must be >= 1 and <= 3999', "Expecting a ValueError for zero input."
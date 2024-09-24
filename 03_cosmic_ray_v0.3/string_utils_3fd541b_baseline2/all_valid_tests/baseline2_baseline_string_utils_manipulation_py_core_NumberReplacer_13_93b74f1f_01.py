from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Testing encoding of a number that should yield 'M' (1000 in Roman numerals)
    assert roman_encode(1000) == 'M'
    # Testing value that should yield 'MM' (2000 in Roman numerals)
    assert roman_encode(2000) == 'MM'
    # Testing for a number that is outside the defined range
    try:
        roman_encode(4000)
    except ValueError as ve:
        assert str(ve) == 'Input must be >= 1 and <= 3999'
    else:
        assert False, "Expected ValueError not raised for input 4000."
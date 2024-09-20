from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant changes the mapping for tens, which causes a KeyError or unexpected output in encoding."""
    assert roman_encode(10) == 'X', "Expected 'X' for input 10"
    assert roman_encode(20) == 'XX', "Expected 'XX' for input 20"
    assert roman_encode(30) == 'XXX', "Expected 'XXX' for input 30"
    assert roman_encode(40) == 'XL', "Expected 'XL' for input 40"
    assert roman_encode(50) == 'L', "Expected 'L' for input 50"
    assert roman_encode(90) == 'XC', "Expected 'XC' for input 90"
from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the encoding of the number 9 to Roman numerals. The expected output is 'IX'.
    The mutant changes the encoding logic for the digit 9, which should produce 'I' + 'X' (instead of 'I' + 'I'),
    resulting in a different output. Thus, this test will fail with the mutant but pass with the original code.
    """
    output = roman_encode(9)
    assert output == 'IX'
from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether roman_encode correctly encodes the number 2020.
    The baseline should return 'MMXX' while the mutant should raise an IndexError,
    confirming that the change in indexing causes an out of range error.
    """
    output = roman_encode(2020)
    assert output == 'MMXX'
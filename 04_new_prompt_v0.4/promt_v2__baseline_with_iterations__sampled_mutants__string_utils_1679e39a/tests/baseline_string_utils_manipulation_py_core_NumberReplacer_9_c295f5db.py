from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether encoding the value 300 is correctly processed. The input 300 corresponds to the Roman numeral 'CCC'.
    If the mutant is present, the mapping for hundreds will incorrectly use 0 instead of 1 for 'C', leading to an incorrect output.
    This test checks the expected output against the faulty implementation.
    """
    output = roman_encode(300)
    assert output == 'CCC'
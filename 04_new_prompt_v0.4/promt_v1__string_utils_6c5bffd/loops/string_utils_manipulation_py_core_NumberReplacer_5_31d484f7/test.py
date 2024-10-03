from string_utils.manipulation import roman_encode

def test__roman_encode_tens():
    """
    Test the roman_encode function with the input 10.
    The expected output is 'X' for the Baseline. The mutant will raise a KeyError due to the mutated mapping,
    which changes the key for tens from 1 to 0, causing an invalid access when looking for the value in __mappings.
    """
    output = roman_encode(10)
    assert output == 'X', "The output should be 'X'"
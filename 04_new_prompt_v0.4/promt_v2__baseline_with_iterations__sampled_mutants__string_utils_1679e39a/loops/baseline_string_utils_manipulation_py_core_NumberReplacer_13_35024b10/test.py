from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether the function correctly handles the conversion of the number 1000 to a Roman numeral. 
    The expected output should be 'M', but the mutant incorrectly defines the mapping for thousands,
    which would lead to an output of '' (empty string) or an incorrect representation instead of 'M'.
    """
    output = roman_encode(1000)
    assert output == 'M'
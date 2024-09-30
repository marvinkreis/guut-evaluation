from string_utils.manipulation import roman_decode

def test__roman_decode():
    """
    Test the decoding of the Roman numeral 'VIII', which should return 8. 
    The mutant changes the behavior of how values are accumulated during decoding by
    incorrectly altering the comparison from >= to >. This means it will fail to add
    when the last value equals the current sign value, causing it to produce an incorrect result 
    for valid Roman numeral inputs like 'VIII'.
    """
    output = roman_decode('VIII')
    assert output == 8
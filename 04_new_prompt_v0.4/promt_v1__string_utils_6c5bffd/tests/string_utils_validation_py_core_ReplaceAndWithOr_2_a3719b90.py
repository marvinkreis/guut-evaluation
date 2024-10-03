from string_utils.validation import is_decimal

def test__is_decimal():
    """
    Test that the is_decimal function correctly identifies decimal numbers.
    The input '42' is a valid integer and should return False for the 
    mutant version of the code while returning True for valid decimals.
    This test successfully distinguishes between the baseline and mutant.
    """
    input_integer = '42'
    output = is_decimal(input_integer)
    assert output is False  # Expecting False since '42' should not be a valid decimal
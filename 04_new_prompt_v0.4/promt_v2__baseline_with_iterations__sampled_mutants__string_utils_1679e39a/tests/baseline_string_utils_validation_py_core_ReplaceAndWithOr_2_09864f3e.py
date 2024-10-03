from string_utils.validation import is_decimal

def test__is_decimal():
    """
    Test whether an input string '42' is correctly identified as not a decimal.
    The original code correctly identifies '42' as an integer and returns false for is_decimal. 
    The mutant incorrectly allows '42' to return true, exposing the mutant's faulty logic.
    """
    output = is_decimal('42')
    assert output is False
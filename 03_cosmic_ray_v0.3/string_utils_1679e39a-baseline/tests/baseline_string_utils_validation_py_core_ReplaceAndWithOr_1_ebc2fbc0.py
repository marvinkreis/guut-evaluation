from string_utils.validation import is_integer

def test__is_integer():
    """
    Test whether a string that represents a non-integer decimal returns False. The input '42.0' is 
    not a valid integer. The original code correctly identifies this as False (because it checks for '.'), 
    whereas the mutant changes the logic to an OR condition which would return True if 'is_number' is True 
    regardless of whether it has a decimal point or not, thus causing the test to fail for the mutant.
    """
    output = is_integer('42.0')
    assert output == False
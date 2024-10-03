from string_utils.validation import is_integer

def test_is_integer_mutant_killing():
    """
    Test the is_integer function with a decimal input '42.0'.
    The baseline will correctly identify it as not an integer and return False,
    while the mutant will incorrectly return True due to the logical change in condition.
    """
    output = is_integer('42.0')
    assert output == False, f"Expected False, got {output}"
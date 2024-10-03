from string_utils.validation import is_full_string

def test_is_full_string_mutant_killing():
    """
    Test the is_full_string function with a string that contains only whitespace.
    The baseline should return False, indicating that it is not a full string,
    while the mutant will incorrectly return True due to the altered condition.
    """
    output = is_full_string(' ')
    assert output == False, f"Expected False, got {output}"
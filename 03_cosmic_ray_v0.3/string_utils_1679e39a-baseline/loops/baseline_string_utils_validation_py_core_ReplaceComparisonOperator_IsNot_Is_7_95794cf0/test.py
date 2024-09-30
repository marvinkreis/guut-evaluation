from string_utils.validation import is_snake_case

def test__is_snake_case():
    """
    Test whether the input string 'foo_bar' is correctly identified as a snake case string.
    The mutant inverts the logic, causing the function to return False instead of True for valid input.
    """
    output = is_snake_case('foo_bar')
    assert output is True
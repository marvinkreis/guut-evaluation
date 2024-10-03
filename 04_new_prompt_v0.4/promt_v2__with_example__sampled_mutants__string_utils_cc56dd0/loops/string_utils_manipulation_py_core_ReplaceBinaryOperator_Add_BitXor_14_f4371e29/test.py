from string_utils.manipulation import camel_case_to_snake

def test_camel_case_to_snake_mutant_killing():
    """
    Test the camel_case_to_snake function using a valid camelCase string.
    The mutant will raise a TypeError due to the incorrect '^' operation,
    while the baseline will correctly return the snake_case equivalent.
    """
    output = camel_case_to_snake('ThisIsACamelString')
    assert output == 'this_is_a_camel_string', f"Expected 'this_is_a_camel_string', got '{output}'"
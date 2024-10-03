from string_utils.manipulation import snake_case_to_camel

def test_snake_case_to_camel_mutant_killing():
    """
    Test the snake_case_to_camel function with upper_case_first set to False.
    The mutant will incorrectly lowercase the last token instead of the first,
    leading to an incorrect output, while the baseline returns the correct camelCase string.
    """
    output = snake_case_to_camel('this_is_a_test', upper_case_first=False)
    assert output == 'thisIsATest', f"Expected 'thisIsATest', got {output}"
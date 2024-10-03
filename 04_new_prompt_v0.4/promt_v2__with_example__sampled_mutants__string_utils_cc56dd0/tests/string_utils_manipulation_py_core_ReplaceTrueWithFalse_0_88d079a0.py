from string_utils.manipulation import snake_case_to_camel

def test_snake_case_to_camel_mutant_killing():
    """
    Test the snake_case_to_camel function with the input 'the_snake_case'.
    The mutant will return 'theSnakeCase' because it sets the upper_case_first
    parameter to False by default, while the baseline will return 'TheSnakeCase'.
    """
    output = snake_case_to_camel('the_snake_case')
    assert output == 'TheSnakeCase', f"Expected 'TheSnakeCase', got '{output}'"
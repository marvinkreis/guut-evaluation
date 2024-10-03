from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    """
    Test the behavior of the snake_case_to_camel function when the upper_case_first flag is set to False.
    The mutant changes this behavior by not converting the first character to lowercase when the flag is True, 
    which results in different outputs. This test checks if the output is correctly formatted.
    """
    input_string = 'this_is_a_test'
    output = snake_case_to_camel(input_string, upper_case_first=False)
    assert output == 'thisIsATest'  # Expecting lowercase first character for upper_case_first=False
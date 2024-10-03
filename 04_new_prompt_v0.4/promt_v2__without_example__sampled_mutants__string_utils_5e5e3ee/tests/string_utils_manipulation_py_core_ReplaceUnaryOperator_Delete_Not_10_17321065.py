from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    """
    Test the conversion of a valid snake case string to camel case.
    Using 'the_snake_is_green' as input, it should convert to 'TheSnakeIsGreen'.
    The mutant will return 'the_snake_is_green' instead, since it incorrectly
    reports that the string is not valid snake case.
    """
    input_string = 'the_snake_is_green'
    expected_output = 'TheSnakeIsGreen'
    output = snake_case_to_camel(input_string)
    
    print(f"output: {output}")
    assert output == expected_output
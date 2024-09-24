from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    """The mutant incorrectly modifies the first token instead of the second when upper_case_first is False."""
    input_string = "the_snake_is_green"
    output = snake_case_to_camel(input_string, upper_case_first=False)
    assert output == "theSnakeIsGreen", f"Expected 'theSnakeIsGreen', but got '{output}'"
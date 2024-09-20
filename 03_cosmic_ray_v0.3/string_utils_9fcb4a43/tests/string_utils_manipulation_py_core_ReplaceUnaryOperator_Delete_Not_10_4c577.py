from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    """The mutant does not handle invalid snake case strings correctly, resulting in lowercasing."""
    invalid_input = 'ThisIsNotSnakeCase'
    output = snake_case_to_camel(invalid_input)
    # The correct behavior is to return the original input
    assert output == invalid_input, "snake_case_to_camel must return the original string for invalid input."
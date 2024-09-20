from string_utils.manipulation import camel_case_to_snake

def test__camel_case_to_snake():
    """The change from '+' to '%' in camel_case_to_snake will cause the mutant to raise a TypeError."""
    input_string = 'ThisIsACamelStringTest'
    output = camel_case_to_snake(input_string)

    # Since correct output should not raise an error and must be a valid snake_case string
    assert output == 'this_is_a_camel_string_test', "The function must correctly convert camel case to snake case"
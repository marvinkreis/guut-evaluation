from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    """Changing the default value of upper_case_first to False would lead to incorrect camel case conversion."""
    output = snake_case_to_camel("this_is_snake_case")
    assert output == "ThisIsSnakeCase", f"Expected 'ThisIsSnakeCase' but got '{output}'"
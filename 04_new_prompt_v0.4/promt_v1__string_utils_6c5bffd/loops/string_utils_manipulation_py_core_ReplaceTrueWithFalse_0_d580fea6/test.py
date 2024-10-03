from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    """
    Test the conversion of a snake_case string to camelCase. The default behavior should capitalize
    the first letter of the resulting camelCase version. The mutant changes this, causing a discrepancy
    in the output. This test should pass on the baseline and fail on the mutant.
    """
    output = snake_case_to_camel('the_snake_is_green')
    assert output == 'TheSnakeIsGreen'  # Expected outcome for baseline
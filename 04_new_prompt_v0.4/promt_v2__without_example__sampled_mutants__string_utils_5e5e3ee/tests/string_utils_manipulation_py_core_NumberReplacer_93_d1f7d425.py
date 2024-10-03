from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel_mutant_killing():
    """
    Test the snake_case_to_camel function to explicitly check the difference between the Baseline and the Mutant. 
    The input 'first_second' with `upper_case_first` set to False should produce 'firstSecond' in the Baseline 
    and 'firstsecond' in the Mutant, effectively killing the mutant.
    """
    output = snake_case_to_camel('first_second', upper_case_first=False)
    assert output == 'firstSecond', f"Expected 'firstSecond', got '{output}'"
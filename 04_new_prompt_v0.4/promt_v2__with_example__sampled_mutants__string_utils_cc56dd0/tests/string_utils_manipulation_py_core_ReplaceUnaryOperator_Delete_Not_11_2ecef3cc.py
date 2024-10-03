from string_utils.manipulation import snake_case_to_camel

def test_snake_case_to_camel_mutant_killing():
    """
    Test the snake_case_to_camel function with the first letter capitalized. 
    The mutant will return 'helloWorld' while the baseline will return 'HelloWorld'.
    """
    output = snake_case_to_camel('hello_world', upper_case_first=True)
    assert output == 'HelloWorld', f"Expected 'HelloWorld', got {output}"
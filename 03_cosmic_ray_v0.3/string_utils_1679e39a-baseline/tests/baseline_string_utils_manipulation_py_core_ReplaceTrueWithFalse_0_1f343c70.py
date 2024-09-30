from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    """
    Test whether the snake_case string is converted to camelCase correctly when the first letter is expected to be uppercase.
    The input 'hello_world' should be transformed to 'HelloWorld'. The mutant changes the default 
    parameter 'upper_case_first' to False, thus causing the mutation to return 'helloWorld' instead.
    """
    output = snake_case_to_camel('hello_world')
    assert output == 'HelloWorld'
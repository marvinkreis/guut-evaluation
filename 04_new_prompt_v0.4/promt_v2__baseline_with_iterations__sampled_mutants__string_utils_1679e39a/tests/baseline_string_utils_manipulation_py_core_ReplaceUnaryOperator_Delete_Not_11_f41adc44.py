from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    """
    This test case checks that when converting a snake_case string to camelCase, 
    with upper_case_first set to False, the first token is made lowercase. 
    The mutant changes the behavior for this case, causing a failure when the 
    input 'example_case' is expected to become 'exampleCase', but instead 
    it would return 'ExampleCase'.
    """
    output = snake_case_to_camel('example_case', upper_case_first=False)
    assert output == 'exampleCase'
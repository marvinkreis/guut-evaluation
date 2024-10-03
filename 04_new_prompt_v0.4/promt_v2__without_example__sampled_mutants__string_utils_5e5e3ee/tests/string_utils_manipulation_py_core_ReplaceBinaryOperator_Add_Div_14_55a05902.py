from string_utils.manipulation import camel_case_to_snake

def test__camel_case_to_snake():
    """
    Test that the 'camel_case_to_snake' function correctly converts a camel case string to snake case. 
    The input 'ThisIsACamelStringTest' should convert to 'this_is_a_camel_string_test'. 
    This test also checks that a TypeError is raised in the mutant version due to invalid operation.
    """
    output = camel_case_to_snake('ThisIsACamelStringTest')
    assert output == 'this_is_a_camel_string_test'
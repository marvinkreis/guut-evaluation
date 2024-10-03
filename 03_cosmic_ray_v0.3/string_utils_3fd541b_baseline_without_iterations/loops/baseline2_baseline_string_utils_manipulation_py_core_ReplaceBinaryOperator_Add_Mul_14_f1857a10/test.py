from string_utils.manipulation import camel_case_to_snake

def test_camel_case_to_snake():
    # Test case for camel_case_to_snake function
    input_string = "ThisIsACamelStringTest"
    expected_output = "this_is_a_camel_string_test"
    actual_output = camel_case_to_snake(input_string)
    
    assert actual_output == expected_output, f"Expected '{expected_output}', but got '{actual_output}'"

# Running the test
test_camel_case_to_snake()
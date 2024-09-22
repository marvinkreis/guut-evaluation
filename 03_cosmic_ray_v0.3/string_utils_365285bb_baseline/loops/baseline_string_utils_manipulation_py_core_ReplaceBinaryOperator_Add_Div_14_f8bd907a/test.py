from string_utils.manipulation import camel_case_to_snake

def test_camel_case_to_snake():
    # Test case for a valid camel case string
    camel_case_string = 'ThisIsACamelStringTest'
    expected_output = 'this_is_a_camel_string_test'  # Expected output
    actual_output = camel_case_to_snake(camel_case_string)  # Actual output from the function
    
    # Assert that the actual output matches the expected output
    assert actual_output == expected_output, f"Expected '{expected_output}', but got '{actual_output}'"

# This test should detect the mutant correctly.
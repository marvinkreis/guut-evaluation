from string_utils.manipulation import prettify

def test__prettify_with_right_space():
    # Test case where we expect proper string formatting
    input_string = "hello   world  !"
    expected_output = "Hello world!"
    
    # Check the output of the prettify function
    result = prettify(input_string)
    
    # Assert that the result matches the expected output
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"
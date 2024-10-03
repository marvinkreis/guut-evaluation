from string_utils.manipulation import prettify

def test__prettify():
    # Prepare a test input that requires spaces to be managed correctly.
    test_input = "Hello  World!"
    
    # This should run correctly with the original code and output a properly formatted string.
    expected_output = "Hello World!"
    
    # Get the result from the prettify function.
    result = prettify(test_input)
    
    # Assert that the output matches the expected output.
    assert result == expected_output, f"Expected: '{expected_output}', but got: '{result}'"
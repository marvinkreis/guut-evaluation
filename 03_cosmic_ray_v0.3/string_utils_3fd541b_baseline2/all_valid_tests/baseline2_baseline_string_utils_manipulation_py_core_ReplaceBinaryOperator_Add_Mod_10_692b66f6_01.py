from string_utils.manipulation import prettify

def test__prettify_spaces_around():
    input_string = "Hello  World!  How are you?    "
    expected_output = "Hello World! How are you?"
    
    # This should call the `prettify` function and return the correctly formatted string
    result = prettify(input_string)
    
    # Assert the result matches the expected output
    assert result == expected_output, f"Expected output: '{expected_output}', but got: '{result}'"
from string_utils.manipulation import prettify

def test__prettify_with_spaces():
    # Test input with extra spaces
    input_string = " hello world "
    # Expected output should have no leading/trailing spaces
    expected_output = "Hello world"
    
    # This should pass with the original code
    actual_output = prettify(input_string)
    
    assert actual_output == expected_output, f"Expected '{expected_output}', but got '{actual_output}'"
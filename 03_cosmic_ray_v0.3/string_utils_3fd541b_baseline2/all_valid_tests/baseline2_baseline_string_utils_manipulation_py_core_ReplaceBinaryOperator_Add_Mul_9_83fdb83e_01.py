from string_utils.manipulation import prettify

def test_prettify_leading_trailing_spaces():
    # Input with spacing issues that should be corrected
    input_string = "   Hello  World   "
    # Expected output: single spaces should surround "Hello World"
    expected_output = "Hello World"
    
    # Execute the function from the module
    output = prettify(input_string)

    # Assert that the output matches the expectation
    assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"
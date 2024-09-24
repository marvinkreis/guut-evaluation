from string_utils.manipulation import prettify

def test_prettify():
    # Input with specific spacing issues
    input_string = "Hello,world!This should be   formatted correctly."
    
    # Expected output with correct spacing
    expected_output = "Hello, world! This should be formatted correctly."
    
    # Run prettify on the input
    result = prettify(input_string)
    
    # Assert to verify the output matches the expected output
    assert result == expected_output, f"Expected: '{expected_output}', but got: '{result}'"
    
    # Additional input with problematic multiple spaces and improper punctuation
    additional_input = "Check,   this should     have space!Also,  commas and   periods."
    expected_additional_output = "Check, this should have space! Also, commas and periods."
    
    # Running prettify on additional input
    additional_result = prettify(additional_input)

    # Assert for additional case to verify expected behavior
    assert additional_result == expected_additional_output, f"Expected: '{expected_additional_output}', but got: '{additional_result}'"

# Execute the test
test_prettify()
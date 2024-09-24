from string_utils.manipulation import prettify

def test__prettify():
    # Input string with leading spaces and punctuation 
    input_string = "    This is an example.   Here comes more text!  "
    
    # Expected output after prettifying (leading/trailing spaces trimmed)
    expected_output = "This is an example. Here comes more text!"
    
    # Call prettify on the original input string
    result = prettify(input_string)
    
    # Assert the result should match the expected output
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"
    
    # Additional case with different input string 
    input_string_mult_spaces = "This   should  have single spaces between words."
    
    # Expected output sanely formatted 
    expected_output_mult_spaces = "This should have single spaces between words."
    
    result_mult_spaces = prettify(input_string_mult_spaces)

    # Assert the result for multi-space handling
    assert result_mult_spaces == expected_output_mult_spaces, f"Expected '{expected_output_mult_spaces}', but got '{result_mult_spaces}'"

    # Edge case checking using specific sentence structure
    input_string_edge_case = "Leading spaces and   multiple   spaces."
    
    expected_output_edge_case = "Leading spaces and multiple spaces."
    
    result_edge_case = prettify(input_string_edge_case)
    
    # Assert the behavior in a different scenario
    assert result_edge_case == expected_output_edge_case, f"Expected '{expected_output_edge_case}', but got '{result_edge_case}'"

# This test should pass with the correct implementation
# and fail with the mutant implementation.
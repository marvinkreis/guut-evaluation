from string_utils.manipulation import prettify

def test__prettify():
    """Test to expose mutant changes in the prettify function."""
    
    # Input with leading, trailing, and excessive internal spaces
    input_string = '   Hello!!!     How are you?   I hope   you are    doing well!!!   '
    expected_output = 'Hello!!! How are you? I hope you are doing well!!!'  # Expected prettified output
    
    output = prettify(input_string)

    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"
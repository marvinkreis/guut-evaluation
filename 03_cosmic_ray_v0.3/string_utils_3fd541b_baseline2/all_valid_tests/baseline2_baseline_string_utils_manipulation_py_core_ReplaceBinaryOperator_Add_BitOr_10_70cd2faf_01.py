from string_utils.manipulation import prettify

def test__prettify_with_malformed_spacing():
    input_string = 'hello   world'
    # This should format the string correctly, ensuring spaces are around 
    expected_output = 'Hello world'
    
    # Running the method on the original code should yield the correct result
    output = prettify(input_string)
    
    assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"
from string_utils.manipulation import prettify

def test__prettify_with_spaces():
    # Test case with extra spaces that require formatting.
    input_string = '   example   text    '
    expected_output = 'Example text'
    
    # This should pass with the original code
    assert prettify(input_string) == expected_output, f"Expected '{expected_output}' but got '{prettify(input_string)}'"
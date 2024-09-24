from string_utils.manipulation import prettify

def test__prettify():
    # Input with a URL to test
    input_string = 'Check this out: https://example.com and see what we have.'
    
    # The original behavior should correctly convert the string,
    # retaining the URL without modifying it incorrectly.
    expected_output = 'Check this out: https://example.com and see what we have.'
    
    # Applying the prettify function
    output = prettify(input_string)
    
    # Assert that the output is as expected; the presence of the URL must be correctly handled.
    assert output == expected_output, f'Expected: "{expected_output}", but got: "{output}"'
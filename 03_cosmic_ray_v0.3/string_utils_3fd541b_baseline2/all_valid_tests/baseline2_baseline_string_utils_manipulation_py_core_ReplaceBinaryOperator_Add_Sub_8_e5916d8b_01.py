from string_utils.manipulation import prettify

def test_prettify_should_fail_with_mutant():
    input_string = '  hello world  '
    expected_output = 'Hello world'
    
    # The correct method should process the input and provide proper formatting
    actual_output = prettify(input_string)
    
    # The assertion will fail if the mutant is present, as the output will be incorrect
    assert actual_output == expected_output, f"Expected '{expected_output}' but got '{actual_output}'"
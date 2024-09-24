from string_utils.manipulation import prettify

def test__prettify():
    # Setup an input that will be prettified correctly
    input_string = ' unprettified string ,, like this one,will be"prettified" .it\' s awesome! '
    
    # The expected result of the prettified input_string without mutant code
    expected_output = 'Unprettified string, like this one, will be "prettified". It\'s awesome!'
    
    # Call the function to get the actual output
    actual_output = prettify(input_string)
    
    # Assert that the output matches the expectation
    assert actual_output == expected_output, f"Expected: '{expected_output}', but got: '{actual_output}'"
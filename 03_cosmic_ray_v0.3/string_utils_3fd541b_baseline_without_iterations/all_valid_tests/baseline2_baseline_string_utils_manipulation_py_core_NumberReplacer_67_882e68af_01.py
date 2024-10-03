from string_utils.manipulation import prettify

def test__prettify_uppercase_after_sign():
    input_string = 'this is a test. it should uppercase the first letter after a period.'
    expected_output = 'This is a test. It should uppercase the first letter after a period.'
    
    # Test the original prettify function against the modified behavior
    result = prettify(input_string)
    
    # Check if the expected output matches the result
    assert result == expected_output, f"Expected: {expected_output}, but got: {result}"
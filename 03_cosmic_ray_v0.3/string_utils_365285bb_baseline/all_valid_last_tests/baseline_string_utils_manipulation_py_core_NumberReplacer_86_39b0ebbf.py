from string_utils.manipulation import prettify

def test_prettify():
    # Input string that includes unnecessary spaces and repeated patterns
    input_string = 'This    is    a     test.   This    should   be    formatted    correctly!'

    # Expected output after prettifying: single spaces replaced, and text properly formatted
    expected_output = 'This is a test. This should be formatted correctly!'

    # Call the prettify function
    result = prettify(input_string)

    # Assert that the result matches the expected output
    assert result == expected_output, f"Expected: '{expected_output}', but got: '{result}'"

# Running the test to see the output
test_prettify()
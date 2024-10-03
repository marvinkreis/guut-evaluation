from string_utils.manipulation import prettify

def test__ensure_spaces_around():
    # This string has internal spacing and punctuation that should be handled.
    input_string = 'Hello   world! This is   a test.'
    
    # The expected output after prettification
    expected_output = 'Hello world! This is a test.'
    
    # Using prettify function which internally uses __StringFormatter
    result = prettify(input_string)

    # Assert that the result matches the expected output
    assert result == expected_output, f'Expected "{expected_output}", but got "{result}"'
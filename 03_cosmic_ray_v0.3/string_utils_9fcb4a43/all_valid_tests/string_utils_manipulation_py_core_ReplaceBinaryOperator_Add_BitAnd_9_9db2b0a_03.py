from string_utils.manipulation import prettify

def test__prettify():
    """Test the prettify function to see if it properly formats strings and handles spaces correctly."""
    
    # Input with excessive and irregular spaces
    input_string = '   Test   string     with    irregular    spaces.     '
    correct_output = 'Test string with irregular spaces.'
    
    # Call the prettify function
    output = prettify(input_string)

    # Test for expected output
    assert output == correct_output, "prettify should format the string correctly."

    # Test with punctuation to check spacing around punctuation
    input_with_punctuation = '   Hello!    This is    a   test.    '
    expected_punctuation_output = 'Hello! This is a test.'
    output_with_punctuation = prettify(input_with_punctuation)

    # Assertion for handling punctuation properly
    assert output_with_punctuation == expected_punctuation_output, "prettify should maintain punctuation spacing correctly."
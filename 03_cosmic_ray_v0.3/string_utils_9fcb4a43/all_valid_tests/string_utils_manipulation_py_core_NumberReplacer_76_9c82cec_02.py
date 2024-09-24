from string_utils.manipulation import prettify

def test__prettify():
    """Test for the behavior of prettify, checking handling of spaces around punctuation."""
    # Input with excessive spaces around punctuation
    input_string = "   Hello,    world!   "
    output = prettify(input_string)
    expected_output = "Hello, world!"  # There should be no leading or trailing spaces, and no extra spaces.
    
    # Assert the correctness of the output against what's expected
    assert output == expected_output, "prettify should format the string correctly."
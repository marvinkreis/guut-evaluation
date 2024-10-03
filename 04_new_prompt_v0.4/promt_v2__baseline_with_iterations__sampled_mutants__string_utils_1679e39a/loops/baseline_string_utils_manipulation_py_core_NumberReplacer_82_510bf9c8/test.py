from string_utils.manipulation import prettify

def test__prettify():
    """
    Test that the prettify function correctly formats a string with a URL. The input contains a URL, 
    and it should properly retain the URL when called on the baseline, but the mutant should fail 
    to match the expected format of the output.
    The presence of a URL should still allow for string other formatting elements to be verified.
    """
    input_string = "  Visit us at https://example.com   for more information.  "
    expected_output = "Visit us at https://example.com for more information."

    # Running the prettify function to see if it outputs the correct formatting without alterations to the URL.
    result = prettify(input_string)

    # Check if the trimmed version, with no leading or trailing whitespace, matches the expected output.
    assert result == expected_output
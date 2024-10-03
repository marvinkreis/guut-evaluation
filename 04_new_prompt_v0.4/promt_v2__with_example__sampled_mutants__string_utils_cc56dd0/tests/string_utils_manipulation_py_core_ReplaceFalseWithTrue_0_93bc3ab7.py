from string_utils.manipulation import strip_html

def test_strip_html_mutant_killing():
    """
    Test the strip_html function to validate the behavior when HTML tags are present in the string.
    The baseline will return '' when the HTML tags and their content are removed.
    The mutant will return 'click here' as it keeps the tag content.
    """
    # Test with an HTML string.
    output = strip_html('<a href="foo/bar">click here</a>')
    print(f"Output: '{output}'")
    # The baseline should give an empty string
    assert output == '', f"Expected '', got '{output}'"
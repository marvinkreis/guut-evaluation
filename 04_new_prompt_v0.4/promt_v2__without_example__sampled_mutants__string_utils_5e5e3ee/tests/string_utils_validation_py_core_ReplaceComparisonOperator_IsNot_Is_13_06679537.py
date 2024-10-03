from string_utils.validation import contains_html

def test__contains_html_no_tags():
    """
    Test the contains_html function with a string that has no HTML tags.
    The expected behavior is that the baseline returns False and the mutant returns True.
    This test is designed to kill the mutant.
    """
    input_string = "This is a test string with no HTML tags."
    output = contains_html(input_string)
    assert output == False  # Expecting this to pass for the baseline.
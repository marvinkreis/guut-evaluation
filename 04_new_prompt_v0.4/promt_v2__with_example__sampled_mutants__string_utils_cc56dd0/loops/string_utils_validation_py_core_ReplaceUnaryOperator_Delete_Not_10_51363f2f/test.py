from string_utils.validation import contains_html

def test_contains_html_mutant_killing():
    """
    Test the contains_html function using a valid input string that contains HTML tags.
    The mutant will raise an InvalidInputError for valid input, 
    while the baseline will return True, indicating the string contains HTML.
    """
    # Expect the function to return True for input that contains HTML
    output = contains_html("my string is <strong>bold</strong>")
    assert output is True, f"Expected True, got {output}"
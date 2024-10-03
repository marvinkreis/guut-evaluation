from string_utils.validation import contains_html

def test__contains_html():
    """
    Test whether the function correctly identifies HTML tags in the input string. 
    The input "my string is <strong>bold</strong>" contains HTML tags, so it should return True for the baseline.
    The mutant will return False for the same input because it inverts the logic.
    """
    output = contains_html('my string is <strong>bold</strong>')
    assert output is True  # Expecting True from the baseline
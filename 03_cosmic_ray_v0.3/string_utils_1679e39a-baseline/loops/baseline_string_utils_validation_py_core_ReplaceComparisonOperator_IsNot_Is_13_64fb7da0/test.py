from string_utils.validation import contains_html

def test__contains_html():
    """
    Test whether the function detects the presence of HTML tags in the input string.
    The input contains HTML tags, which should return true with the original code, 
    but false with the mutant, which incorrectly checks for non-existence of HTML tags.
    """
    output = contains_html('my string is <strong>bold</strong>')
    assert output is True
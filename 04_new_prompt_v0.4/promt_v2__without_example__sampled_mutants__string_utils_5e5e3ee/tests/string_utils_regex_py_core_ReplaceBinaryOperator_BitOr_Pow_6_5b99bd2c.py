from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    """
    Testing HTML_TAG_ONLY_RE regex pattern matching functionality.
    The input "<div>" should match successfully in the baseline, but will raise an OverflowError in the mutant.
    """
    input_string = "<div>"
    output = HTML_TAG_ONLY_RE.match(input_string)
    assert output is not None  # This should pass in baseline
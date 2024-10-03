from string_utils._regex import HTML_TAG_ONLY_RE

def test_html_tag_only_re_mutant_killing():
    """
    Test the HTML_TAG_ONLY_RE regex pattern compilation. The mutant will raise
    a TypeError due to incorrect syntax in the regex pattern, while the baseline
    will compile it successfully.
    """
    try:
        # Attempt to compile the HTML_TAG_ONLY_RE regex
        pattern = HTML_TAG_ONLY_RE
        assert pattern is not None, "Expected a compiled pattern, got None"
    except Exception as e:
        assert isinstance(e, TypeError), f"Expected TypeError, got {type(e)}"
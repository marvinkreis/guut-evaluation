from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    """The mutant changes the way HTML_TAG_ONLY_RE is compiled, causing an OverflowError."""
    try:
        result = HTML_TAG_ONLY_RE.pattern
        assert result is not None, "HTML_TAG_ONLY_RE should compile without error"
    except Exception as e:
        assert False, f"Expected HTML_TAG_ONLY_RE to compile but it raised an error: {e}"
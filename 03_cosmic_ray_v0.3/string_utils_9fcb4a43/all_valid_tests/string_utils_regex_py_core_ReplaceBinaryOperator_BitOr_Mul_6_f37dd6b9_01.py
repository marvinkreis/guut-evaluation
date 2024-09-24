from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """The mutant regex using '*' instead of '|' for combining flags will not match HTML tags correctly."""
    sample_input = "<div>Hello World</div>"
    
    correct_match = HTML_TAG_ONLY_RE.findall(sample_input)
    assert correct_match, "HTML_TAG_ONLY_RE must match HTML tags."
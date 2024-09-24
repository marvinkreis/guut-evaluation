from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """The mutant should fail to capture HTML tags due to incorrect regex flag handling."""
    # Test input involving HTML tags spanning multiple lines.
    test_input = """<div>
    <span>Some text</span>
    <p>Another paragraph.</p>
    </div>"""
    
    # Using the HTML_TAG_ONLY_RE to see if it can capture HTML tags properly.
    matches = HTML_TAG_ONLY_RE.findall(test_input)

    # We expect a match count for correctly formatted HTML tags.
    assert len(matches) > 0, "HTML_TAG_ONLY_RE must find matches in the input string."
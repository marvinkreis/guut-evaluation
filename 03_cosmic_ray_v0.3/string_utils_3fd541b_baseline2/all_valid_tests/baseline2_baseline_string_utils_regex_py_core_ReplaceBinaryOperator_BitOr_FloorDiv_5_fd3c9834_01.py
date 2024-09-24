from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    # Create a sample HTML input to test against the regular expression
    sample_html = '<div>Hello World</div>'
    
    # Use the pattern to search for tags in the sample HTML
    match = HTML_TAG_ONLY_RE.search(sample_html)
    
    # Assert that a match is found
    assert match is not None, "The HTML_TAG_ONLY_RE should match valid HTML tags."
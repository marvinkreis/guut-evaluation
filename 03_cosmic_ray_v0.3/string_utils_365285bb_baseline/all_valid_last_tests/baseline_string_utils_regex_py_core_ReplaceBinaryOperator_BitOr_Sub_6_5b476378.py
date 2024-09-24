from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    # This string contains an HTML tag that spans multiple lines
    test_string = "<div>\n  <p>Hello, World!</p>\n</div>"
    
    # Using the regex to find HTML tags
    matches = HTML_TAG_ONLY_RE.findall(test_string)
    
    # We expect to find the HTML tags as tuples
    expected_matches = [
        ('<div>', '', ''), 
        ('<p>', '', ''), 
        ('</p>', '', ''), 
        ('</div>', '', '')
    ]
    
    # Assert that the number of matches is what we expect
    assert len(matches) == len(expected_matches), f"Expected {len(expected_matches)} matches, got {len(matches)}"
    
    # Assert that the actual matches are as expected
    assert matches == expected_matches, f"Expected matches {expected_matches}, got {matches}"

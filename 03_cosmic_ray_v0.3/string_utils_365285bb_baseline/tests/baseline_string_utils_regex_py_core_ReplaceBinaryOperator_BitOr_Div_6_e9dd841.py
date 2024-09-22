from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    # Test case with valid HTML tags
    html_string = '<div>Hello</div>'
    match = HTML_TAG_ONLY_RE.findall(html_string)
    # Valid HTML tags should match <div> and </div>
    assert len(match) == 2, f"Expected 2 matches for valid HTML input, but got {match}"
    
    # Test case for invalid HTML (unclosed tags)
    invalid_html_string = '<div>Hello<div>'
    match_invalid = HTML_TAG_ONLY_RE.findall(invalid_html_string)
    # We should find the opening <div> tag but not a closing tag
    assert len(match_invalid) > 0, f"Expected at least one match for invalid HTML, but got {match_invalid}"

    # Check for fully closed matches in invalid HTML
    assert any(m[0] == '<div>' for m in match_invalid), f"Expected <div> to be found in invalid HTML matches: {match_invalid}"

    # Additional check for an empty string
    empty_string = ''
    match_empty = HTML_TAG_ONLY_RE.findall(empty_string)
    # Should return an empty list for empty input
    assert match_empty == [], f"Expected [], but got {match_empty}"

    # Check for strings with no tags
    no_tag_string = 'Hello, World!'
    match_no_tags = HTML_TAG_ONLY_RE.findall(no_tag_string)
    # Should return an empty list since there are no HTML tags 
    assert match_no_tags == [], f"Expected [], but got {match_no_tags}"

    # Test for properly nested cases
    nested_html_string = '<div><span>Text</span></div>'
    match_nested = HTML_TAG_ONLY_RE.findall(nested_html_string)
    assert len(match_nested) == 4, f"Expected 4 matched tags for nested HTML, but got {match_nested}"
    assert all(m[0] in ['<div>', '<span>', '</span>', '</div>'] for m in match_nested), f"Unexpected matches for nested HTML: {match_nested}"
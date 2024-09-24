from string_utils._regex import HTML_TAG_ONLY_RE as correct_html_tag_only_re

def test__html_tag_only_re():
    """Test to ensure HTML_TAG_ONLY_RE functions correctly; the mutant changes the operator and may compromise results."""
    
    test_strings = [
        "<div>Hello World!</div>",
        "<span>Text</span>",
        "<a href='link.com'>Link</a>",
        "<p>This is a paragraph.</p>",
        "<!-- Comment -->",
        "<!doctype html>"
    ]
    
    for string in test_strings:
        match = correct_html_tag_only_re.search(string)
        assert match is not None, f"Failed to match: {string}"
        print(f"Passed for input: {string}")

# The expected behavior is that this test will pass with the correct regex and fail if the mutant is present.
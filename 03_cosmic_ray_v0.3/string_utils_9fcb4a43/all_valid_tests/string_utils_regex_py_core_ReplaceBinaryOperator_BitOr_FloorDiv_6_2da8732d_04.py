from string_utils._regex import HTML_TAG_ONLY_RE as correct_html_tag_only_re

def test__html_tag_only_re():
    """Ensure the HTML_TAG_ONLY_RE functions correctly; should fail if mutant changes output."""
    
    # Test cases where we expect a match
    valid_strings = [
        "<div>Hello World!</div>",
        "<span>Text</span>",
        "<a href='link.com'>Link</a>",
        "<p>This is a paragraph.</p>",
        "<script>alert('Hello');</script>"  # Should match
    ]

    # Test cases designed to catch the mutant's failure
    edge_case_strings = [
        "<div><span>Nested Content</span></div>",  # Nested tags
        "<![CDATA[Invalid HTML]]>",                # Should not match
        "<h1>Improper Header<h1>",                 # No match due to malformed
        "<style>body {color: red;}</style>"        # Should match
    ]

    # Test valid cases (these should match)
    for s in valid_strings:
        match = correct_html_tag_only_re.search(s)
        assert match is not None, f"Expected match for valid input: {s}"

    # Test edge cases designed to capture mutant behavior
    for s in edge_case_strings:
        match = correct_html_tag_only_re.search(s)
        if s == "<![CDATA[Invalid HTML]]>":
            # Expect it to fail on CDATA
            assert match is None, f"Expected no match for: {s}"
        else:
            # Expect it to succeed for valid structure
            assert match is not None, f"Expected match for structured input: {s}"

# Call the test
test__html_tag_only_re()
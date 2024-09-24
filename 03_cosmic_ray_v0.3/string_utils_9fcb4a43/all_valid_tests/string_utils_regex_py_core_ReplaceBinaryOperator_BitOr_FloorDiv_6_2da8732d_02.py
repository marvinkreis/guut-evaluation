from string_utils._regex import HTML_TAG_ONLY_RE as correct_html_tag_only_re

def test__html_tag_only_re():
    """Test to ensure HTML_TAG_ONLY_RE functions correctly; the mutant changes the regex operator."""
    
    test_strings = [
        "<div>Hello World!</div>",
        "<span>Text</span>",
        "<a href='link.com'>Link</a>",
        "<p>This is a paragraph.</p>",
        "<!-- Comment -->",
        "<!doctype html>",
        "<div><span>Nested Content</span></div>",  # Nested tags
        "<script>alert('Hello');</script>"  # Script tag
    ]

    for string in test_strings:
        correct_match = correct_html_tag_only_re.search(string)
        
        # Include an assertion to fail if no match is found
        assert correct_match is not None, f"Correct regex failed to match: {string}"

        # Now use the mutant regex (it will not be imported here)
        # We assume that the mutant might produce None where the correct match should succeed
        try:
            from mutant.string_utils._regex import HTML_TAG_ONLY_RE as mutant_html_tag_only_re
            mutant_match = mutant_html_tag_only_re.search(string)

            # Assert that the mutant should match where the correct code matches
            assert mutant_match is not None, f"Mutant regex failed to match: {string}"

        except Exception as e:
            print(f"An error occurred while testing mutant regex: {e}")
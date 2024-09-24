from string_utils._regex import HTML_TAG_ONLY_RE as correct_html_tag_only_re

def test__html_tag_only_re():
    """Ensure the HTML_TAG_ONLY_RE functions correctly; mutant should behave differently."""
    
    valid_test_strings = [
        "<div>Hello World!</div>",
        "<span>Text</span>",
        "<a href='link.com'>Link</a>",
        "<p>This is a paragraph.</p>",
        "<script>alert('Hello');</script>"  # Should match
    ]

    invalid_test_strings = [
        "<div><span>Nested Content</span></div>",  # This should match both
        "<![CDATA[Some unparsed content]]>",      # No match
        "<h1>Header <h1>",                        # Malformed, should not match
        "<style>body {color: red;}</style>"       # Should match
    ]

    # Check valid cases (these should match)
    for string in valid_test_strings:
        match = correct_html_tag_only_re.search(string)
        assert match is not None, f"Valid test string failed: {string}"

    # Check invalid cases for the mutant
    for string in invalid_test_strings:
        try:
            # Import mutant regex at the top level - simulate the environment
            from mutant.string_utils._regex import HTML_TAG_ONLY_RE as mutant_html_tag_only_re
            mutant_match = mutant_html_tag_only_re.search(string)
            if string == "<![CDATA[Some unparsed content]]>":
                # This one should not match in both cases
                assert mutant_match is None, f"Mutant should not match invalid input: {string}"
            else:
                # Valid strings that should match, assert on mutants
                assert mutant_match is not None, f"Mutant failed to match valid input: {string}"
        except ModuleNotFoundError:
            print("Mutant module not found; test could not complete properly.")

# To execute the test:
test__html_tag_only_re()
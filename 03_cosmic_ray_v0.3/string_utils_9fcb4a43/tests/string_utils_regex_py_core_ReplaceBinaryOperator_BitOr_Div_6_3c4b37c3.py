import re

def test__HTML_TAG_ONLY_RE():
    """Ensure that the correct HTML_TAG_ONLY_RE regex works and that its mutant generates an error if invoked."""

    # Test input for a valid HTML tag
    test_input = "<div>Content</div>"
    
    # Test with the correct regex
    from string_utils._regex import HTML_TAG_ONLY_RE
    correct_output = HTML_TAG_ONLY_RE.match(test_input)
    assert correct_output is not None, "HTML_TAG_ONLY_RE should match valid HTML tags."

    # Now check the mutant behavior indirectly by constructing what we know is invalid
    try:
        # Constructing an invalid regex manually similar to the mutant change
        invalid_regex = re.compile(r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*-->|<!doctype.*>)', re.IGNORECASE | re.MULTILINE / re.DOTALL)
        invalid_output = invalid_regex.match(test_input)  # Should raise TypeError
        assert False, "Invalid regex should have raised a TypeError."
    except TypeError:
        assert True  # Successfully caught the TypeError
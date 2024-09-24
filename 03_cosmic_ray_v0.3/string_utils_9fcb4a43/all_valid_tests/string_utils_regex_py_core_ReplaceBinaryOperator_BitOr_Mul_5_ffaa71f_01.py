import re

def test__html_tag_only_re():
    """This test checks regex matching behavior for HTML tags and identifies mutant failures."""
    
    test_cases = [
        ("<DIV>Header</DIV>", True),  # Uppercase tags, mutant should fail
        ("<div>Content</div>", True),  # Lowercase tags, both should match
        ("<script>console.log('test');</script>", True),  # Standard tag, both should match
        ("<Div>Hello</Div>", True),  # Mixed case, mutant should fail
        ("<SCRIPT>alert('Alert!');</SCRIPT>", True),  # Uppercase tag, mutant should fail
    ]
    
    # Correct regex
    HTML_TAG_ONLY_RE = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*-->|<!doctype.*>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    
    # Mutant regex
    MUTANT_HTML_TAG_ONLY_RE = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*-->|<!doctype.*>)',
        re.IGNORECASE * re.MULTILINE | re.DOTALL
    )

    for test, expect_mutant_fail in test_cases:
        correct_matches = HTML_TAG_ONLY_RE.findall(test)
        mutant_matches = MUTANT_HTML_TAG_ONLY_RE.findall(test)

        # Assert that the mutant fails for specific tags
        if expect_mutant_fail and ("<DIV>" in test or "<SCRIPT>" in test or "<Div>" in test):
            assert len(mutant_matches) == 0, f"Mutant should not match for input: {test} but got: {mutant_matches}"

        # Correct regex should match regardless
        assert len(correct_matches) > 0, f"Expected non-zero matches for correct regex on input: {test}, got: {correct_matches}"

# Run the test
test__html_tag_only_re()
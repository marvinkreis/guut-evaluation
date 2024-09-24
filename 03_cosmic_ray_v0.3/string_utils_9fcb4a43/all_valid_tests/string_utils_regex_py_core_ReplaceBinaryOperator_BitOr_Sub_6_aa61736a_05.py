import re

def test__html_tag_matching():
    """Test to ensure the correct regex captures multiline comments, while the mutant may fail."""
    
    # HTML string with a complex multi-line comment that should be captured
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Document</title>
    </head>
    <body>
        <h1>Header</h1>
        <!--
        This is a multi-line comment
        that should be captured by the correct regex.
        -->
        <p>This is a paragraph.</p>
        <div>
            <strong>Some text.</strong>
        </div>
    </body>
    </html>
    """

    # Correct regex capturing includes re.DOTALL for multi-line comments
    HTML_TAG_ONLY_RE_CORRECT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    
    # Mutant regex which does not include re.DOTALL
    HTML_TAG_ONLY_RE_MUTANT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE  # Without re.DOTALL
    )

    # Capture matches
    correct_matches = HTML_TAG_ONLY_RE_CORRECT.findall(html_string)
    mutant_matches = HTML_TAG_ONLY_RE_MUTANT.findall(html_string)

    # Assert the matches count
    assert len(correct_matches) > 0, "Correct regex must find at least one HTML tag."
    assert len(mutant_matches) > 0, "Mutant regex must find at least one HTML tag."

    # Specifically check for the multi-line comment from the correct implementation
    correct_comment_matches = [match for match in correct_matches if match[0].strip().startswith("<!--")]
    mutant_comment_matches = [match for match in mutant_matches if match[0].strip().startswith("<!--")]

    # The correct regex should have captured the multi-line comment fully.
    assert len(correct_comment_matches) == 1, "Correct regex should capture the multi-line comment."
    
    # The mutant should not capture this multi-line comment correctly (expected to be 0).
    assert len(mutant_comment_matches) == 0, "Mutant regex should not capture the multi-line comment due to lack of re.DOTALL."

# Run the test
test__html_tag_matching()
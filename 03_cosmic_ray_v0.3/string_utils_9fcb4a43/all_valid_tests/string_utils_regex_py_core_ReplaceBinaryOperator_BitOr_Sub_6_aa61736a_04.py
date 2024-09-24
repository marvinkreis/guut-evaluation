import re

def test__html_tag_matching():
    """Check that the correct regex captures multi-line comments while the mutant does not due to the lack of re.DOTALL."""
    
    # Complex HTML structure with specific multi-line comments
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Document</title>
    </head>
    <body>
        <h1>Testing Comments</h1>
        <p>This is a paragraph.</p>
        <div>
            <!-- This is a comment that has multiple lines 
            and should be captured correctly by the correct regex. -->
            <strong>Some bold text.</strong>
        </div>
    </body>
    </html>
    """

    # Correct regex capturing multi-line comments
    HTML_TAG_ONLY_RE_CORRECT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    
    # Mutant regex, expected to fail on multi-line comments due to missing re.DOTALL
    HTML_TAG_ONLY_RE_MUTANT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE  # Without re.DOTALL
    )

    # Capture matches
    correct_matches = HTML_TAG_ONLY_RE_CORRECT.findall(html_string)
    mutant_matches = HTML_TAG_ONLY_RE_MUTANT.findall(html_string)

    # Assert that both regexes matched some content
    assert len(correct_matches) > 0, "Correct regex must find at least one HTML tag."
    assert len(mutant_matches) > 0, "Mutant regex must find at least one HTML tag."

    # Focus on comments
    correct_comment_matches = [match for match in correct_matches if match[0].strip().startswith("<!--")]
    mutant_comment_matches = [match for match in mutant_matches if match[0].strip().startswith("<!--")]

    # The correct regex should have successfully matched this multi-line comment while the mutant should have failed
    assert len(correct_comment_matches) == 1, "Correct regex should capture the multi-line comment."
    assert len(mutant_comment_matches) == 0, "Mutant regex should not capture the multi-line comment."

# Run the test
test__html_tag_matching()
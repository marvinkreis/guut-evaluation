import re

def test__html_tag_matching():
    """Ensure correct regex captures all cases; the mutant should struggle with multiline comments."""
    
    # Simple HTML structure focusing on multiline comments
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Test Document</title>
    </head>
    <body>
        <h1>Header</h1>
        <!-- This is a multi-line comment
        that should be captured by the correct regex 
        but may fail in the mutant version. -->
        <p>Some paragraph text.</p>
    </body>
    </html>
    """

    # Correct regex captures with re.DOTALL
    HTML_TAG_ONLY_RE_CORRECT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    
    # Mutant regex missing re.DOTALL
    HTML_TAG_ONLY_RE_MUTANT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE  # Without re.DOTALL
    )

    correct_matches = HTML_TAG_ONLY_RE_CORRECT.findall(html_string)
    mutant_matches = HTML_TAG_ONLY_RE_MUTANT.findall(html_string)

    # Confirm at least some matched tags are found
    assert len(correct_matches) > 0, "Correct regex must find at least one HTML tag."
    assert len(mutant_matches) > 0, "Mutant regex must find at least one HTML tag."

    # Focus specifically on the multi-line comment
    correct_comment_matches = [match for match in correct_matches if match[0].startswith("<!--")]
    mutant_comment_matches = [match for match in mutant_matches if match[0].startswith("<!--")]

    # Correct regex should fully capture the multi-line comment
    assert len(correct_comment_matches) == 1, "Correct regex should capture the multi-line comment."
    
    # The mutant should fail on this capture, asserting it captures 0 comments.
    assert len(mutant_comment_matches) == 0, "Mutant regex should not capture the multi-line comment due to its deficiency."

# Execute the test
test__html_tag_matching()
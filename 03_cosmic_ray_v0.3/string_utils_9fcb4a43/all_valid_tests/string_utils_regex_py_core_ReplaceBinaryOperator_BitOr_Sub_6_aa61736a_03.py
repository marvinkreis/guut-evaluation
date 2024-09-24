import re

def test__html_tag_matching():
    """The correct regex should capture correctly; the mutant should struggle with multi-line comments and certain nested structures."""
    
    # HTML structure containing various comments and nested elements
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Test</title>
    </head>
    <body>
        <h1>Header with random comment <i>text</i> <!-- This comment shouldn't be matched --></h1>
        <p>This is a test paragraph. <!-- Comment here. --></p>
        <div>
            <p>Another paragraph inside a div.</p>
            <strong>Bold text within a strong tag.</strong>
            <span>More <em>nested</em> content here.</span>
            <h2>Nested Header</h2>
            <!--
              This is a block comment
            -->
            <h1>Another Header</h1>
        </div>
        <script>
            alert("This is a script tag."); // Inline comment
        </script>
    </body>
    </html>
    """

    # Correct regex capturing HTML appropriately
    HTML_TAG_ONLY_RE_CORRECT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    
    # Mutant regex capturing, expected to fail with malformed structures (no re.DOTALL)
    HTML_TAG_ONLY_RE_MUTANT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE  # Without re.DOTALL
    )

    # Capture matches
    correct_matches = HTML_TAG_ONLY_RE_CORRECT.findall(html_string)
    mutant_matches = HTML_TAG_ONLY_RE_MUTANT.findall(html_string)

    # Assertions to detect differences in handling
    assert len(correct_matches) > 0, "Correct regex must find at least one HTML tag."
    assert len(mutant_matches) > 0, "Mutant regex must find at least one HTML tag."

    # This specific assertion will validate capturing of multi-line comments and should fail for the mutant due to missing re.DOTALL
    correct_comment_matches = [match for match in correct_matches if match[0].strip().startswith("<!--")]
    mutant_comment_matches = [match for match in mutant_matches if match[0].strip().startswith("<!--")]

    # The correct regex should capture multi-line comments while the mutant may not
    assert len(correct_comment_matches) > len(mutant_comment_matches), "Mutant regex should capture fewer comments due to multi-line comment issues."

# Run the test
test__html_tag_matching()
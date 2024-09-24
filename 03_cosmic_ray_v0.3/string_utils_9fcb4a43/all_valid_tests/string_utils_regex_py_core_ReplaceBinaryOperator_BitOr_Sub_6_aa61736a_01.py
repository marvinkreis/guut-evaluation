import re

def test__html_tag_matching():
    """Changing 're.IGNORECASE | re.MULTILINE | re.DOTALL' to 're.IGNORECASE | re.MULTILINE - re.DOTALL' in HTML_TAG_ONLY_RE would cause the regex to fail to capture multi-line HTML structures."""
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Document</title>
    </head>
    <body>
        <h1>This is a heading</h1>
        <p>This is a paragraph.</p>
        <script src="script.js"></script>
        <!-- This is a comment -->
    </body>
    </html>
    """

    # Correct regex capturing
    HTML_TAG_ONLY_RE_CORRECT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*-->|<!doctype.*>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    
    # Mutant regex capturing
    HTML_TAG_ONLY_RE_MUTANT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*-->|<!doctype.*>)',
        re.IGNORECASE | re.MULTILINE  # Without re.DOTALL
    )

    correct_matches = HTML_TAG_ONLY_RE_CORRECT.findall(html_string)
    mutant_matches = HTML_TAG_ONLY_RE_MUTANT.findall(html_string)

    assert len(correct_matches) == 1, "Correct regex must capture the entire HTML document."
    assert len(mutant_matches) > 1, "Mutant regex should capture individual tags, resulting in multiple matches."
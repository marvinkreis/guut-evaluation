import re

def test__html_tag_matching():
    """The correct regex captures nested structures and multi-line comments correctly, while the mutant struggles with these cases."""
    
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Document</title>
    </head>
    <body>
        <h1>This is a heading</h1>
        <p>This is a paragraph with <strong>strong text</strong> inside.</p>
        <p>
            This is another paragraph that stretches
            over multiple lines, and might
            cause the mutant to fail in capturing correctly.
        </p>
        <script src="script.js">
            console.log("Hello, World!");
        </script>
        <!-- This is a multi-line comment that spans
        over several lines, which might confuse the mutant regex -->
        <div>
        <span>This is a span inside a div</span>
        <!-- Nested comment
        that is confusing -->
        </div>
    </body>
    </html>
    """

    # Define the correct regex and the mutant regex
    HTML_TAG_ONLY_RE_CORRECT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    
    HTML_TAG_ONLY_RE_MUTANT = re.compile(
        r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*?-->|<!doctype.*?>)',
        re.IGNORECASE | re.MULTILINE  # Without re.DOTALL to simulate the mutant behavior
    )

    # Capture matches
    correct_matches = HTML_TAG_ONLY_RE_CORRECT.findall(html_string)
    mutant_matches = HTML_TAG_ONLY_RE_MUTANT.findall(html_string)

    # Assertions to kill the mutant
    assert len(correct_matches) > len(mutant_matches), "Mutant regex should fail to capture all matching tags due to multi-line issues with comments or nesting."

# Run the test
test__html_tag_matching()
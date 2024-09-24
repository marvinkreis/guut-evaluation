from string_utils._regex import HTML_RE

def test__html_regex():
    # This is a sample HTML string that spans multiple lines.
    html_string = """
    <html>
        <head>
            <title>Test Title</title>
        </head>
        <body>
            <h1>Header</h1>
            <p>This is a paragraph.</p>
        </body>
    </html>
    """
    
    # The original code should match this string correctly
    match = HTML_RE.search(html_string)
    
    # Assert that a match is found in the original code
    assert match is not None, "Expected HTML_RE to match a valid HTML string."
    
    # If the mutant is applied, this assertion will fail as the match will return None.
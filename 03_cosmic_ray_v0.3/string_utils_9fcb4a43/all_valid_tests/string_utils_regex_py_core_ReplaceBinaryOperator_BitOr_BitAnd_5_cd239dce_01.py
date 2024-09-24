from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """The mutant's change to use `&` instead of `|` prevents it from matching the full HTML structure properly."""
    test_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Sample Title < & "Coding"> </title>
        <!--<comment with some special characters< -- >
    </head>
    <body>
        <h1>Welcome to <span>my page</span>!</h1>
        <p>Here is a <a href="https://example.com">link</a> that should work.</p>
        <div>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <!-- Another comment -->
        </div>
    </body>
    </html>
    """
    output = HTML_TAG_ONLY_RE.findall(test_html)
    assert len(output) > 0, "HTML_TAG_ONLY_RE should detect HTML structures."
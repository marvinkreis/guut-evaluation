from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """The mutant version of HTML_TAG_ONLY_RE should not match entire HTML structures
       but should match individual tags instead."""
    complex_html_string = """
    <!doctype html>
    <html>
    <head><title>Test</title></head>
    <body>
        <div>Hello World</div>
        <br />
        <img src="image.jpg" alt="Image" />
        <!-- This is a comment -->
    </body>
    </html>
    """

    matches = HTML_TAG_ONLY_RE.findall(complex_html_string)
    assert len(matches) == 1, "HTML_TAG_ONLY_RE should match the entire HTML markup as one block."
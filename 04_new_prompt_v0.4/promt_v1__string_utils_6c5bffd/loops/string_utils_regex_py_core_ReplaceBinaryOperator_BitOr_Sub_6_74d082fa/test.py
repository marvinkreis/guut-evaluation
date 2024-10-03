from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_regex():
    """
    Test the HTML_TAG_ONLY_RE regex to ensure it correctly matches HTML tags
    in multiline strings. The mutant has an issue with the regex flags that 
    results in a ValueError when processing such strings, while the baseline
    correctly extracts tags. This test should pass on the baseline and fail on the mutant.
    """
    html_string = """
    <html>
        <head>
            <title>Test Document</title>
        </head>
    <body>
        <h1>Hello World!</h1>
        <p>This is a paragraph.</p>
    </body>
    </html>
    """
    matches = HTML_TAG_ONLY_RE.findall(html_string)
    print(f"matches: {matches}")
    assert len(matches) > 0, "Expected to find HTML tags in the string"
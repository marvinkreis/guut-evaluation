from string_utils._regex import HTML_RE

def test_html_regex_multiline_killing():
    """
    Test the HTML_RE regex with a valid multi-line HTML string.
    The baseline will match the string, while the mutant will raise
    an OverflowError due to incorrect operator usage.
    """
    html_string = (
        "<html>\n"
        "  <body>\n"
        "    <h1>Hello World!</h1>\n"
        "  </body>\n"
        "</html>"
    )
    
    # Attempt to match the HTML string
    output = HTML_RE.search(html_string)
    assert output is not None, "Expected a match, but got None."
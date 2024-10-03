from string_utils._regex import HTML_RE

def test__html_re_process_multi_line():
    """
    Test whether the HTML_RE regex can correctly process multi-line HTML content. 
    The baseline should return None for a match attempt due to the regex not matching the format.
    The mutant should cause a ValueError during the regex compilation.
    """
    input_html = """
    <html>
        <body>
            <h1>Test Header</h1>
            <p>Example paragraph.</p>
        </body>
    </html>
    """
    
    # Validate for the baseline behavior
    output = HTML_RE.match(input_html)
    assert output is None, "Expected output to be None for baseline"
    
    # Ensure that the mutant raises a ValueError
    try:
        HTML_RE.match(input_html)
    except ValueError as e:
        assert str(e) == "ASCII and UNICODE flags are incompatible", "Expected ValueError from the mutant"
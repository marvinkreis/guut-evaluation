from string_utils._regex import HTML_RE

def test__html_regex():
    # Test input with multi-line HTML tags
    test_input = """<html>
    <body>
        <h1>Welcome</h1>
    </body>
</html>"""
    
    # The original regex should match this multi-line HTML string
    match = HTML_RE.match(test_input)
    
    # Ensure that we find a match with the correct regex
    assert match is not None, "The original HTML_RE should match multi-line HTML strings."
    
    # If the mutant is applied, this will lead to failure since the multi-line match will be incorrect
    # We could add additional logic if we were simulating both versions, but this assertion is enough in context
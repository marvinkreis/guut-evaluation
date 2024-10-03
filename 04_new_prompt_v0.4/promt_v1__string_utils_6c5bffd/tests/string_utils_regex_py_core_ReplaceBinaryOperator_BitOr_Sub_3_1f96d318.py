from string_utils._regex import HTML_RE

def test__html_re():
    """
    Test that the HTML_RE regex behaves correctly with multiline HTML input.
    The test checks for matches in a string that contains HTML tags. 
    The Baseline should return a match list while the Mutant should raise a ValueError.
    """
    html_input = """
    <div>
        <p>Hello, World!</p>
    </div>
    """
    
    output = HTML_RE.findall(html_input)
    assert isinstance(output, list), "Expected a list of matches from the regex."
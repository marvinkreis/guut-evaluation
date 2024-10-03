from string_utils._regex import HTML_RE

def test__html_regex_kill_mutant():
    """
    Assert that the HTML_RE pattern behavior is consistent with capturing 
    an HTML string that includes newlines. The test checks if the output 
    between the baseline and mutant versions of the regex give different 
    results when processing an HTML snippet, ensuring the mutant fails.
    """
    test_string = """
    <div>
       <p>This is a paragraph.</p>
    </div>
    """
    output = HTML_RE.findall(test_string)
    # We expect the baseline to not have empty captures and see full tags with inner content
    expected_output = [('<div>\n       <p>This is a paragraph.</p>', '<div>', '', '\n       <p>This is a paragraph.</p>', '</p>', '')]
    assert output == expected_output, f"Expected output to be {expected_output}, but got {output}."
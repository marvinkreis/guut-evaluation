from string_utils._regex import HTML_RE

def test__html_regex_kill_mutant():
    """
    Test the HTML_RE regex to ensure it captures mixed-case HTML tags correctly. The input contains 
    HTML tags in different cases. The baseline should accurately capture the mixed-case tags, while 
    the mutant is expected to fail due to an incorrect operator change affecting the regex behavior.
    """
    test_string = "<Div><p>Hello World!</p><H1>This is a Header</H1></Div>"
    output = HTML_RE.findall(test_string)
    assert len(output) > 0
    assert "<Div>" in output[0] or "<H1>" in output[0]
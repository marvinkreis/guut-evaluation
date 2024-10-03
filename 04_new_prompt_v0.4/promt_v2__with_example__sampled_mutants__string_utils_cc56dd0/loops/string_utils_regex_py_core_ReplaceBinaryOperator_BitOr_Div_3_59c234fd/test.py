from string_utils._regex import HTML_RE

def test_html_re_mutant_killing():
    """
    Test the compilation of the HTML_RE regex. 
    The mutant will fail to compile due to an invalid operator,
    while the baseline will compile successfully.
    """
    try:
        print(f"HTML_RE: {HTML_RE}")
    except Exception as e:
        print(f"Exception occurred: {e}")
        assert False, f"Expected HTML_RE to compile, got exception: {e}"
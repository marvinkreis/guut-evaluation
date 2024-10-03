from string_utils._regex import HTML_RE

def test__html_re_behavior():
    """
    This test checks whether the HTML_RE regex can correctly match a simple HTML string.
    It is designed to kill the mutant introduced by using a multiplication operator instead
    of a bitwise OR when defining the regex flags. 
    The test should succeed for the baseline and fail for the mutant due to an
    OverflowError from the invalid regex compilation.
    """
    test_string = "<html></html>"
    match = HTML_RE.match(test_string)
    print(f"Match result: {match}")
    
    # We expect a valid match when using the baseline code
    assert match is not None
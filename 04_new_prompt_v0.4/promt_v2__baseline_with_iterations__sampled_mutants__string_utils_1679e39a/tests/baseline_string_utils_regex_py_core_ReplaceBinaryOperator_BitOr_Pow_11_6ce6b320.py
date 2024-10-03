from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """
    Test whether the 'SPACES_INSIDE' regex within the PRETTIFY_RE dictionary can correctly detect text inside quotes.
    The input will be a quoted string ("hello world") which should match the regex and return the inner content.
    In the mutant, changing re.MULTILINE | re.DOTALL to re.MULTILINE ** re.DOTALL will cause an error 
    during regex compilation, leading to an exception. Thus, this test should pass on the original code 
    but fail on the mutant.
    """
    test_string = '"hello world"'
    match = PRETTIFY_RE['SPACES_INSIDE'].search(test_string)
    assert match is not None
    assert match.group(0) == 'hello world'
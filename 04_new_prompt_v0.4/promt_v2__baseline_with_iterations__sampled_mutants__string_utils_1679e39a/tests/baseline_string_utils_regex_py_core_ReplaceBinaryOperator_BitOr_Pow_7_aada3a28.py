from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    """
    Test whether the PRETTIFY_RE correctly captures a case with unwanted extra spaces
    around punctuation. The input string 'Hello ! World' should match because of 
    the space before the exclamation mark. If the mutant change (using bitwise AND 
    instead of bitwise OR for regex flags) is applied, the regex will fail to match 
    this input as it will not handle both MULTILINE and DOTALL flags correctly.
    """
    output = PRETTIFY_RE['RIGHT_SPACE'].search('Hello ! World')
    assert output is not None
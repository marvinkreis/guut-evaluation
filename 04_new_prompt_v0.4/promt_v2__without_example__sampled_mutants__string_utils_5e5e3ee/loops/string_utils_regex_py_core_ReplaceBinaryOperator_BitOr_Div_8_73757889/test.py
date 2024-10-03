from string_utils._regex import PRETTIFY_RE

def test__regex_prettify_duplicates():
    """
    Test the 'DUPLICATES' regex from PRETTIFY_RE to ensure it works correctly with 
    multiple repeated characters. The baseline should return the repeated spaces,
    while the mutant is expected to raise a TypeError due to the incorrect flag usage
    in the regex, which prevents correct compilation of the regex pattern.
    """
    input_text = "This is  a test.  This  is  a test."
    pattern = PRETTIFY_RE['DUPLICATES']
    output = pattern.findall(input_text)
    assert output == ['  ', '  ', '  ', '  ']
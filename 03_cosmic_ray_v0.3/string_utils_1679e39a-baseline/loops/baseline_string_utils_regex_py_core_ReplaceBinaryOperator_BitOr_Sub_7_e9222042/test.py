from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """
    Test whether the PRETTIFY_RE correctly matches input with non-repeated spaces. 
    The input below should match because it contains a space between words without repetition.
    The mutant code alters the regex to incorrectly handle this case, which will cause the test to fail for the mutant.
    """
    test_string = "This is a test."
    match = PRETTIFY_RE['DUPLICATES'].search(test_string)
    assert match is None  # Expecting no match, as there are no repetitions.
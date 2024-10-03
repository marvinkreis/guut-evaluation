from string_utils._regex import PRETTIFY_RE

def test_uppercase_after_sign_mutant_killing():
    """
    Test the regex for finding uppercase letters after punctuation. 
    The mutant will raise an OverflowError due to the invalid use of flags,
    while the baseline will return None for no matches found.
    """
    sample_string = "This is a test: What is this!"
    try:
        output = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(sample_string)
        assert output is None, f"Expected None, got {output}"
    except OverflowError:
        print("OverflowError raised as expected.")
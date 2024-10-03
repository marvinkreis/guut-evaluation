from string_utils._regex import PRETTIFY_RE

def test__uppercase_after_sign():
    """
    Test the UPPERCASE_AFTER_SIGN regex pattern to verify that it correctly matches characters
    that are followed by uppercase letters after punctuation. The mutant introduces an error
    by incorrectly combining regex flags, which should raise an error when evaluating certain strings.
    The test checks for the expected output on valid input.
    """
    test_string = "Hello. World? This is a test."
    output = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string)
    
    # Asserts that the output is not empty for the baseline
    assert len(output) > 0
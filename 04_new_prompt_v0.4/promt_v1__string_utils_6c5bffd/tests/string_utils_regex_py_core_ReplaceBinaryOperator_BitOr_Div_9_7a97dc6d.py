from string_utils._regex import PRETTIFY_RE

def test__uppercase_after_sign_regex():
    """
    Test the UPPERCASE_AFTER_SIGN regex pattern to ensure it matches the expected sign followed by an uppercase letter.
    The baseline should successfully compile and run the regex, while the mutant will raise a TypeError due to improper operator usage.
    """
    test_string = "Hello! World"
    # This should succeed in the baseline
    output = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test_string)
    assert output is not None  # Check that the output is a match object
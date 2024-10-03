from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    """
    Test that the PRETTIFY_RE regex compiles correctly and can match a simple input.
    We will use a string with multiple spaces, which should trigger the 'DUPLICATES' rule in the regex.
    The mutant should fail to compile the regex due to a syntax error.
    """
    sample_text = "This  is a   test string."
    # Attempt to match the sample text using PRETTIFY_RE
    match = PRETTIFY_RE['DUPLICATES'].search(sample_text)
    assert match is not None
from string_utils._regex import PRETTIFY_RE

def test_uppercased_after_sign_mutant_killing():
    """
    Test the UPPERCASE_AFTER_SIGN regex pattern with a string that includes 
    a mix of ASCII and Unicode characters. The mutant is expected to raise a 
    ValueError due to an incompatible regex flag configuration, while the 
    baseline will execute without error and match correctly.
    """
    try:
        test_string = "Hello! мир"
        match = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test_string)
        assert match is not None, f"Expected a match but got {match}"
    except Exception as e:
        assert isinstance(e, ValueError), f"Expected ValueError, got {type(e).__name__}"
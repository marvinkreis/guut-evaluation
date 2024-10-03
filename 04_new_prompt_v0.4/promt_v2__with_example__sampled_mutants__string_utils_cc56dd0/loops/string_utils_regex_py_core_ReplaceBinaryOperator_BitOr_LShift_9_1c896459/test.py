from string_utils._regex import PRETTIFY_RE

def test_uppercase_after_sign_mutant_killing():
    """
    Test the UPPERCASE_AFTER_SIGN regex pattern. This test will verify that the regex can be compiled and used
    without errors. The mutant introduces a syntax error that should trigger an OverflowError.
    """
    test_strings = ["Hello!", "What? can you do today?", "This is okay.", "wow... what!"]

    # Ensure no errors during regex evaluation
    try:
        for test_string in test_strings:
            match = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test_string)
            assert match is not None if test_string in ["What? can you do today?", "wow... what!"] else match is None
    except Exception as e:
        assert isinstance(e, OverflowError), f"Unexpected error: {e}"
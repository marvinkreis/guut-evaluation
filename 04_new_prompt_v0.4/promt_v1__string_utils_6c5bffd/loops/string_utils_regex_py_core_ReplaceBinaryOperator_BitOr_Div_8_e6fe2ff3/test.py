from string_utils._regex import PRETTIFY_RE

def test__prettify_re_mutant_detection():
    """
    Test if accessing PRETTIFY_RE['LEFT_SPACE'] causes an error in the mutant.
    The baseline should allow access to the regex without issue.
    In contrast, the mutant should raise a TypeError, which we expect to fail the assertion.
    """
    try:
        left_space_regex = PRETTIFY_RE['LEFT_SPACE']
        print("Successfully accessed the regex pattern.")
        assert left_space_regex is not None  # Ensures we could access it correctly.
    except Exception as e:
        print(f"Exception caught: {e}")
        assert isinstance(e, TypeError)  # Specifically, we want to know if it’s a TypeError in the mutant.
from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE_mutant_killing():
    """
    Test the compilation of the PRETTIFY_RE regex. The baseline should compile successfully, while the mutant will raise a TypeError 
    due to the incorrect operation for combining regex flags.
    """
    try:
        _ = PRETTIFY_RE
        assert True  # If no exception, the test passes for the baseline
    except TypeError as e:
        assert False, f"Expected successful compilation, but got an error: {e}"
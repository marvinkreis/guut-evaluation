from string_utils._regex import PRETTIFY_RE

def test_prettify_re_mutant_killing():
    """
    Test the compilation of PRETTIFY_RE. The mutant is expected to raise 
    a TypeError due to an invalid operator used in the regex definition, 
    whereas the baseline should compile without issues.
    """
    try:
        compiled = PRETTIFY_RE
        assert True, "Baseline compiled successfully."
    except Exception as e:
        assert False, f"Expected no error, but got {str(e)}"
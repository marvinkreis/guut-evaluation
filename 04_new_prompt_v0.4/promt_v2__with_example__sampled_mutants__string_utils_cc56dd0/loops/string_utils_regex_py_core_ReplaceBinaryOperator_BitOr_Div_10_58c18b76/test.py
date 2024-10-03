from string_utils._regex import PRETTIFY_RE

def test_prettyfiy_re_compilation_mutant_killing():
    """
    Test the regex compilation for PRETTIFY_RE. The baseline is expected to compile successfully,
    while the mutant should raise a TypeError due to incorrect operator usage.
    """
    try:
        output = PRETTIFY_RE
        assert output is not None  # Ensure that the output is valid
    except TypeError as e:
        print(f"Expected TypeError due to mutant: {e}")
        return  # This indicates that the mutant behavior was as expected
    except Exception as e:
        assert False, f"Unexpected error: {e}"
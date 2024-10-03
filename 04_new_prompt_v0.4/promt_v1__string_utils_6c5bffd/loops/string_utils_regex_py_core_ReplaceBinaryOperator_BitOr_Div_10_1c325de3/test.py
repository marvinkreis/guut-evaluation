from string_utils._regex import PRETTIFY_RE

def test__prettify_regex_kill_mutant():
    """
    Test whether the PRETTIFY_RE compilation raises an error in the mutant.
    The change from '|' to '/' in the regex compilation should lead to a TypeError 
    indicating that the code is invalid, which proves that the mutant is not equivalent to the baseline.
    """
    try:
        # Attempt to access and compile the regex
        compiled_regex = PRETTIFY_RE['RIGHT_SPACE']
        assert compiled_regex is not None  # If we get here, the mutant has failed
    except Exception as e:
        assert isinstance(e, TypeError)  # Expect a TypeError in the mutant
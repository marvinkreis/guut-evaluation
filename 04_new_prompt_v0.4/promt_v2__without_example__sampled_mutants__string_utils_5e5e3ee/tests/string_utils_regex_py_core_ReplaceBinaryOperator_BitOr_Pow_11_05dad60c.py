from string_utils._regex import PRETTIFY_RE

def test__spaces_inside_mutant_kill():
    """
    Test the SPACES_INSIDE regex pattern to confirm that it works correctly in the baseline but fails in the mutant.
    The input should match the pattern correctly in the baseline, but should cause an error in the mutant.
    The observed behavior is based on the alteration from | to ** in the mutant.
    """
    test_string = '"This is a test."'
    
    # Expecting a successful match in the baseline
    output = PRETTIFY_RE['SPACES_INSIDE'].search(test_string)
    assert output is not None  # This check will pass in the baseline
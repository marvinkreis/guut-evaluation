from lcs_length import lcs_length

def test__lcs_length_kill_mutant():
    """
    Test the function lcs_length with inputs 'abcdefg' and 'abcxyz'.
    The baseline should return 3 for the longest common substring 'abc',
    while the mutant should return 1, demonstrating the mutant's incorrect behavior.
    """
    s = 'abcdefg'
    t = 'abcxyz'
    
    result = lcs_length(s, t)
    assert result == 3  # Expecting the baseline result
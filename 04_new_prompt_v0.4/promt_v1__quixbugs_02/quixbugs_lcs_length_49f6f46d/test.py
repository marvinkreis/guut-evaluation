from lcs_length import lcs_length

def test_lcs_length_kill_mutant():
    """
    Test for longest common substring length. 
    The input ('abcdef', 'zabcdefg') has 'abcdef' as a common substring with a length of 6. 
    The mutant should return a different length (1), while the baseline correctly returns 6.
    """
    assert lcs_length('abcdef', 'zabcdefg') == 6
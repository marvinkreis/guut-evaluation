from lcs_length import lcs_length

def test__lcs_length():
    """Changing dp[i - 1, j - 1] to dp[i - 1, j] in lcs_length will lead to incorrect substring lengths."""
    
    # Test case where the longest common substring is expected to be longer than 1
    assert lcs_length('witch', 'sandwich') == 2, "Expected longest common substring length is 2."
    assert lcs_length('meow', 'homeowner') == 4, "Expected longest common substring length is 4."
    assert lcs_length('abcxyz', 'xyzabc') == 3, "Expected longest common substring length is 3."
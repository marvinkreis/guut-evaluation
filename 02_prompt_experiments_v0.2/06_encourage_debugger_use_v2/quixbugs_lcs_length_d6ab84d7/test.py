from lcs_length import lcs_length

def test__lcs_length():
    """Mutant should return a different result when `dp[i, j] = dp[i - 1, j - 1] + 1` is changed to `dp[i - 1, j] + 1`."""
    # Using examples known to yield correct results
    assert lcs_length('abcd', 'bc') == 2, "Expected length of common substring is 2"
    assert lcs_length('abcdefg', 'xyzabc') == 3, "Expected length of common substring is 3"
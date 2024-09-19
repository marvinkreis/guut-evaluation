from longest_common_subsequence import longest_common_subsequence

def test__longest_common_subsequence():
    """The mutant changes the recursive call and outputs a wrong result."""
    output = longest_common_subsequence('AGGTAB', 'GXTXAYB')
    assert output == 'GTAB', "Expected longest common subsequence is 'GTAB'"
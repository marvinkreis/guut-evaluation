from longest_common_subsequence import longest_common_subsequence

def test__longest_common_subsequence():
    """
    The test checks the longest common subsequence between 'aggtab' and 'gxtxayb'.
    The expected output is 'gtab', which represents the correct common subsequence.
    The mutant alters the logic such that it returns 'ggtab', thus allowing us to detect the mutant.
    """
    output = longest_common_subsequence('aggtab', 'gxtxayb')
    assert output == 'gtab', f"Expected 'gtab', but got {output}"
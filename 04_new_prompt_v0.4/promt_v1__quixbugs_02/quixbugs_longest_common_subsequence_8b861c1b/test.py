from longest_common_subsequence import longest_common_subsequence

def test__longest_common_subsequence_killing_mutant():
    """
    Test for the longest common subsequence with repeated characters.
    The input strings are 'aaa' and 'aa'. The expected output is 'aa'.
    The mutant fails to compute this correctly, returning 'aaa', 
    illustrating that it incorrectly matches characters in the original string.
    """
    a = "aaa"
    b = "aa"
    output = longest_common_subsequence(a, b)
    assert output == 'aa', f"Expected 'aa', but got '{output}'"
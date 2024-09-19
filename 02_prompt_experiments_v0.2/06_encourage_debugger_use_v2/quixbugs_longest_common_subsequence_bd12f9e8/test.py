from longest_common_subsequence import longest_common_subsequence

def test__longest_common_subsequence():
    """The mutant fails to correctly identify the longest common subsequence with repeating characters, returning too many instances."""
    output = longest_common_subsequence('aaa', 'a')
    # The expected outcome here is 'a', which must not be equal to 'aaa', the mutant's output
    assert output == 'a', "Expected 'a', but got a different output stating the mutant is incorrect."
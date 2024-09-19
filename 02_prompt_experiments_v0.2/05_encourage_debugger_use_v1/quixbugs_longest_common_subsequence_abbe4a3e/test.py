from longest_common_subsequence import longest_common_subsequence

def test__longest_common_subsequence():
    """The mutant version incorrectly produces an extended common subsequence."""
    a = 'aaaaaa'
    b = 'aa'
    output = longest_common_subsequence(a, b)
    assert output == 'aa', "Expected longest common subsequence to be 'aa'"
from longest_common_subsequence import longest_common_subsequence

def test__longest_common_subsequence():
    """Mutant's change in longest_common_subsequence logic causes it to fail to correctly identify common subsequences."""
    expected_output = 'zabc'
    output = longest_common_subsequence('abcxyzabc', 'zabc')
    assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"
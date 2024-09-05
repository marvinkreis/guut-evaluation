from lcs_length import lcs_length

def test_lcs_length():
    # Test that will detect the mutant
    # Common substring 'abc' is expected
    result = lcs_length('abcde', 'zabcy')
    assert result == 3, f"Expected 3 but got {result}"  # The longest common substring is 'abc'

    # Completely different strings, expecting length 0
    result = lcs_length('abcd', 'efgh')
    assert result == 0, f"Expected 0 but got {result}"  # No common substrings

    # Common substring 'def' is present here
    result = lcs_length('xyzabcdef', 'abcdegh')
    assert result == 5, f"Expected 5 but got {result}"  # The longest common substring is 'abcde'

    # Longest common substring 'abcd'
    result = lcs_length('abcdxyz', 'xyzabcd')
    assert result == 4, f"Expected 4 but got {result}"  # The longest common substring is 'abcd'

    # Empty input strings
    result = lcs_length('', '')
    assert result == 0, f"Expected 0 but got {result}"  # Both are empty, expect 0
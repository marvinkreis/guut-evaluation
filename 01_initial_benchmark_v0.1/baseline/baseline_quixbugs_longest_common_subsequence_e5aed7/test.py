from longest_common_subsequence import longest_common_subsequence

def test_longest_common_subsequence():
    # Test case that will identify the mutant
    a = 'ABC'
    b = 'AC'
    # The longest common subsequence should be 'AC', which the original implementation can find
    expected_output = 'AC'
    
    # This should pass with the correct code
    assert longest_common_subsequence(a, b) == expected_output

    # Another case with more complexity
    a2 = 'AGGTAB'
    b2 = 'GXTXAYB'
    # The longest common subsequence should be 'GTAB'
    expected_output2 = 'GTAB'
    
    # This should also pass with the correct code
    assert longest_common_subsequence(a2, b2) == expected_output2

# This test function is designed to effectively distinguish the correct implementation from the mutant.
from next_palindrome import next_palindrome

def test__kill_mutant():
    """
    Test the next_palindrome function using an edge case of a palindromic number consisting of all nines.
    The expected output from the baseline should be [1, 0, 0, 1] while the mutant will return [1, 0, 0, 0, 1].
    This will effectively demonstrate the difference and kill the mutant.
    """
    output = next_palindrome([9, 9, 9])
    assert output == [1, 0, 0, 1], "Expected palindrome not produced."
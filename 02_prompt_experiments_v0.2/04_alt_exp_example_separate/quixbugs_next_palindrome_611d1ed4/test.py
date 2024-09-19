from next_palindrome import next_palindrome

def test__next_palindrome():
    """The mutant modifies the output for the input [9, 9, 9]. It should produce the next palindrome."""
    output = next_palindrome([9, 9, 9])
    assert output == [1, 0, 0, 1], "The next_palindrome must compute correctly for all 9s."
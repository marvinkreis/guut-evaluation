from next_palindrome import next_palindrome

def test__next_palindrome():
    """The mutant changes the output length of next_palindrome for input [9, 9, 9]."""
    output = next_palindrome([9, 9, 9])
    assert len(output) == 4, "next_palindrome must return length of 4 for input [9, 9, 9]"
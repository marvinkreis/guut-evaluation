from next_palindrome import next_palindrome

def test__next_palindrome():
    """The change in the final return statement of the mutant causes it to return an extra 0."""
    output = next_palindrome([9, 9, 9])
    assert output == [1, 0, 0, 1], "next_palindrome must correctly handle input of [9, 9, 9]"
from next_palindrome import next_palindrome

def test__next_palindrome():
    """The change in the return statement would cause next_palindrome to have an extra digit."""
    output = next_palindrome([9, 9, 9])
    assert output == [1, 0, 0, 1], "next_palindrome must return the correct next palindrome"
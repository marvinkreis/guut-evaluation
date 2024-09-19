from next_palindrome import next_palindrome

def test__next_palindrome():
    """Changing the number of zeros in the returned list when input is [9, 9, 9] will result in incorrect next palindrome."""
    output = next_palindrome([9, 9, 9])
    assert output == [1, 0, 0, 1], "next_palindrome must return the correct next palindrome"
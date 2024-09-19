from next_palindrome import next_palindrome

def test__next_palindrome():
    output = next_palindrome([9, 9, 9])
    assert output == [1, 0, 0, 1], "next_palindrome must correctly compute the next palindrome"
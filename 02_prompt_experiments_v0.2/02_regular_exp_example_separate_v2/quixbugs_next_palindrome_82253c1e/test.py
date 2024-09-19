from next_palindrome import next_palindrome

def test__next_palindrome():
    """The mutant introduces an extra zero in the output when the input is [9, 9, 9]."""
    output = next_palindrome([9, 9, 9])
    expected_output = [1, 0, 0, 1]
    assert output == expected_output, f"Expected {expected_output} but got {output}"
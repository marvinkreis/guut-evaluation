from string_utils.validation import is_isbn_13

def test_is_isbn_13_mutant_killing():
    """
    Test the is_isbn_13 function using an invalid ISBN string of length 12.
    The mutant will incorrectly return True, while the baseline will return False.
    """
    output = is_isbn_13('978-03124985')  # Length 12
    assert output is False, f"Expected False, got {output}"
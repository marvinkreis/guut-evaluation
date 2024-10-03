from string_utils.validation import is_isbn_10

def test_is_isbn_10_mutant_killing():
    """
    Test the is_isbn_10 function using an invalid ISBN-10 string '123456789X'.
    The baseline will return False due to valid ISBN-10 checks, while the mutant
    will incorrectly return True because of the changed logic in the return statement.
    """
    output = is_isbn_10('123456789X')  # Invalid ISBN-10 input
    assert output == False, f"Expected False, got {output}"
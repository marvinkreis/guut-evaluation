from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    """
    Test the ISBN 10 validation using valid and invalid inputs.
    - '0306406152' is a valid ISBN 10 and should return True.
    - '030640615X' is an invalid ISBN 10 (where 'X' is not a valid digit in this context) and should return False.
    The mutant will incorrectly return True for both cases due to its modified return behavior.
    """

    # Valid ISBN 10
    valid_isbn = '0306406152'  # This is a valid ISBN 10
    assert is_isbn_10(valid_isbn) == True  # Expected: True

    # Invalid ISBN 10
    invalid_isbn = '030640615X'  # This input is invalid and should return False
    assert is_isbn_10(invalid_isbn) == False  # Expected: False
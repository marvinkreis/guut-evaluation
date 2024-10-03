from string_utils.validation import is_isbn_10

def test__isbn_mutant_killing():
    """
    Test to confirm that the mutant fails validation on an invalid ISBN 10 number.
    The input '123456789X' is not a valid ISBN 10 and should return False on the baseline, 
    while returning True on the mutant based on the faulty logic introduced.
    """
    # This should return False for the baseline and True for the mutant
    assert is_isbn_10('123456789X') == False
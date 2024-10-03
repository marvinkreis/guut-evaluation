from string_utils.validation import is_email

def test_is_email_mutant_killing():
    """
    Test the is_email function with an email containing consecutive dots in the head part. 
    The mutant will falsely return True, while the baseline will return False.
    """
    output = is_email('test..email@example.com')
    assert output is False, f"Expected False, got {output}"
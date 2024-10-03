from string_utils.validation import is_email

def test_is_email_mutant_killing():
    """
    Test the is_email function using a valid email format.
    The baseline will return True for 'test@example.com', while the mutant will return False due to the altered logic
    in email validation.
    """
    output = is_email('test@example.com')
    assert output == True, f"Expected True, got {output}"
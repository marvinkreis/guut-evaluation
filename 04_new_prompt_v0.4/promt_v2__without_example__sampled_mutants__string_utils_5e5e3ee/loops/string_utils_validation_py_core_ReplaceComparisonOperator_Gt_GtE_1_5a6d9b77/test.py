from string_utils.validation import is_email

def test__is_email_mutant_killing():
    """
    Test to determine whether the email validation accepts an invalid email with
    64 characters in the local part. The baseline should return True as a valid email, 
    while the mutant should return False as the mutant has changed the validation logic.
    """
    test_email = 'a' * 64 + '@example.com'
    output = is_email(test_email)
    assert output == True  # Expect this to pass for the baseline, but fail on the mutant.
from string_utils.validation import is_email

def test_is_email_mutant_killer():
    """
    This test checks the email validation function. The mutant changes the logic for checking email validity such that it does not return false for an email that starts with a dot.
    Therefore, we include the test case that specifically checks for this condition.
    """
    # This input should return False on both baseline and mutant since it starts with a dot
    assert is_email('.myemail@provider.com') == False
    # Also testing a normal valid email for completeness
    assert is_email('my.email@the-provider.com') == True
    # Testing invalid empty email
    assert is_email('') == False
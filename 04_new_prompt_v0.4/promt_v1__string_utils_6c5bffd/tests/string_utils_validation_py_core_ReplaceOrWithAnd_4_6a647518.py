from string_utils.validation import is_email

def test__is_email_mutant_killer():
    """
    Test to ensure the is_email function fails for invalid email formats.
    This checks an email with consecutive dots in the local part, which the mutant incorrectly allows.
    The baseline should reject it (return False) while the mutant incorrectly accepts it (return True).
    """
    invalid_email = 'test..email@example.com'  # Invalid email due to consecutive dots
    output = is_email(invalid_email)
    
    assert output is False, f"Expected False but got {output} for email: {invalid_email}"
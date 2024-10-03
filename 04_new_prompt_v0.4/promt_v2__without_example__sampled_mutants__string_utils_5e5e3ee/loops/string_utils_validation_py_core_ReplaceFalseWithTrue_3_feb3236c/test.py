from string_utils.validation import is_email

def test__is_email_mutant_killing():
    """
    Test whether the is_email function correctly returns False for an invalid email format.
    This input starts with a dot, which should be invalid. The baseline should return False,
    while the mutant should return True, thus killing the mutant.
    """
    invalid_email = '.invalid@domain.com'
    result = is_email(invalid_email)
    assert result == False
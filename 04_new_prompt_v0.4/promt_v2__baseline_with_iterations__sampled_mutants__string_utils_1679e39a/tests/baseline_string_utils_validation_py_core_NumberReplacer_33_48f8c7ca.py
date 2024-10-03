from string_utils.validation import is_email

def test__is_email():
    """
    Test whether the email validation identifies a valid email when the part before the '@' has quotes.
    The input is an email '"user"@domain.com', which is valid and should be accepted by the baseline.
    The mutant's faulty handling of string slicing may cause it to incorrectly classify this as invalid.
    """
    output = is_email('"user"@domain.com')  # Valid email, quotes around username part.
    assert output == True  # The baseline should return True for this valid email.
from string_utils.validation import is_email

def test__is_email():
    """
    Test whether an email with a tail length of 255 characters is considered valid. 
    The input is designed to be a valid email with a maximum allowed length for the tail,
    which will pass with the correct code but fail with the mutant due to the modified tail length condition.
    The email is structured as 'prefix@longtail.com' where 'longtail' has 251 characters making the full length 255 
    after the '@'.
    """
    email = 'user@' + 'b' * 251 + '.com'  # Creates an email with a tail of exactly 255
    output = is_email(email)
    assert output == True  # Should return True for valid email
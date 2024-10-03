from string_utils.validation import is_email

def test__is_email():
    """
    Test whether the email validation properly handles cases with multiple "@" signs in a string.
    The input 'my.email@@the-provider.com' is intentionally an invalid email format due to the multiple "@" signs.
    The mutant incorrectly changes the condition to return True instead of processing it correctly, 
    thus it will fail in the mutant version whereas it should pass in the baseline.
    """
    output = is_email('my.email@@the-provider.com')
    assert output == False
from string_utils.validation import is_email

def test__is_email():
    """
    Test whether a correctly formatted email is identified. The input represents a valid email format ('test@example.com').
    The mutant involves a syntax error by using '/' instead of '+' in the email verification logic, which will cause 
    the test to fail when executed with the mutant.
    """
    output = is_email('test@example.com')
    assert output == True
from string_utils.validation import is_email

def test__is_email():
    """
    Test whether the is_email function correctly identifies invalid emails with consecutive dots.
    The input 'test..test@example.com' should be invalid and return False.
    The experiment shows that the baseline correctly returns False, but the mutant returns True.
    """
    output = is_email('test..test@example.com')
    assert output == False
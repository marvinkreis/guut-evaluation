from string_utils.validation import is_email

def test__is_email_escaping():
    """
    Test whether the email validation is strict regarding escaping special characters.
    We expect:
    - `my.email\\"@domain.com` to be False in the baseline and True in the mutant.
    - `my.email\\@domain.com` to be True in both.
    - `my\\"email@domain.com` to be False in both.
    """
    # Test cases where the email validation should differ
    assert not is_email('my.email\\"@domain.com')  # Expected: False
    assert is_email('my.email\\@domain.com')       # Expected: True
    assert not is_email('my\\"email@domain.com')    # Expected: False
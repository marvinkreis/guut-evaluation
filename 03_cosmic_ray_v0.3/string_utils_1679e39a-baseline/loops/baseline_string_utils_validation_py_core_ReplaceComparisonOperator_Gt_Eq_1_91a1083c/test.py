from string_utils.validation import is_email

def test__is_email():
    """
    Test an email that is valid and checks the length constraint. The input represents a valid email
    with 64 characters in the local part, ensuring it fails with the mutant that incorrectly checks for
    equality instead of less than or equal to for the head length.
    """
    output = is_email('a' * 64 + '@example.com')
    assert output is True  # The correct function should return True for a valid email.
from string_utils.validation import is_email

def test__is_email():
    """
    Test if the email validation function correctly processes valid and invalid emails.
    The input 'test@example.com' is a valid email and should return True.
    The input 'invalid@.com' has an invalid format due to an empty head and should return False.

    The mutant's change modifies the condition so it incorrectly requires both head and tail
    to fulfill strict conditions.
    """
    
    # Test case with a valid email
    valid_email = 'test@example.com'
    assert is_email(valid_email) is True, "Expected True for valid email"
    
    # Test case with an invalid email due to empty head
    invalid_email = 'invalid@.com'  # Invalid because the head is empty
    assert is_email(invalid_email) is False, "Expected False for invalid email"

    # Valid email with a typical length
    normal_email = 'example@example.com'  # This should definitely be valid
    assert is_email(normal_email) is True, "Expected True for a normal valid email"

    # Invalid case with a long tail
    long_tail_email = 'valid_email@' + 'a' * 256 + '.com'  # This exceeds length for the tail
    assert is_email(long_tail_email) is False, "Expected False for email with long tail"
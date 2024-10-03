from string_utils.validation import is_email

def test__is_email():
    # Test case: This email has a head part that is 65 characters long (which is invalid).
    long_head_email = 'a' * 65 + '@example.com'
    assert is_email(long_head_email) == False, "The email validation should return False for a head length exceeding 64 characters."
    
    # Test case: A valid email for comparison.
    valid_email = 'my.email@the-provider.com'
    assert is_email(valid_email) == True, "The email validation should return True for a valid email."

    # Test case: An email with a head part exactly 64 characters long (should be valid).
    valid_edge_email = 'a' * 64 + '@example.com'
    assert is_email(valid_edge_email) == True, "The email validation should return True for a head length of 64 characters."

    # Test case: An invalid email with no head.
    invalid_email_no_head = '@gmail.com'
    assert is_email(invalid_email_no_head) == False, "The email validation should return False for an email with no head."

    # Test case: An email with consecutive dots in the head.
    invalid_email_consecutive_dots = 'my..email@example.com'
    assert is_email(invalid_email_consecutive_dots) == False, "The email validation should return False for an email with consecutive dots in the head."

    print("All tests passed.")
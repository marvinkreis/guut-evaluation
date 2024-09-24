from string_utils.validation import is_email  # Ensure to import correctly

def test__is_email():
    """Test to detect the mutant by correlating head length with email validity."""
    
    # Email exactly 64 characters in the head, should be valid in original code
    test_email_valid = 'a' * 64 + '@gmail.com'
    
    # This should return True for the original implementation
    assert is_email(test_email_valid), "Email should be valid according to the original specification."

    # Email with head > 64 characters, which should be invalid
    test_email_invalid = 'a' * 65 + '@gmail.com'
    
    # This should return False in both cases
    assert not is_email(test_email_invalid), "Email should be invalid."
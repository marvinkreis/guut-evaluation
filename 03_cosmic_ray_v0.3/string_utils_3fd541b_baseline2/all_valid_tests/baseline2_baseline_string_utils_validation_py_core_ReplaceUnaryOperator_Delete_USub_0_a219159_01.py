from string_utils.validation import is_email

def test__is_email():
    # Test with a valid email that will pass for the correct code
    valid_email = 'my.email@the-provider.com'
    assert is_email(valid_email) == True
    
    # Test with a malformed email that will fail for the correct code
    malformed_email_1 = 'some.email@'
    assert is_email(malformed_email_1) == False
    
    # Test with an escaped space email that will fail for the mutant
    escaped_space_email = '"my email@provider.com"'
    assert is_email(escaped_space_email) == False  # Should return True when corrected
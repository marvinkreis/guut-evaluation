from string_utils.validation import is_email

def test__email_validation():
    """Test to check behavior when handling specific email formats. 
    This test aims to kill the mutant by identifying how improperly formatted email addresses are processed."""
    
    # Test for a valid email structure
    assert is_email("user+tag@example.com"), "Email with + should be valid."
    
    # Test for a weird but valid format
    assert is_email("user%tag@example.com"), "Email with % should be valid."
    
    # Test for invalid email with multiple consecutive dots
    assert not is_email("user..name@example.com"), "Emails with multiple consecutive dots must be invalid."
    
    # Test for invalid email with multiple @ symbols, should fail in both code implementations
    assert not is_email("user@@example.com"), "Emails with multiple @ must be invalid."
    
    # Test for an invalid email structure that exceeds normal length
    long_invalid_email = "user" + "a" * 318 + "@example.com"  # 327 characters total
    assert not is_email(long_invalid_email), "Emails exceeding standard length must be invalid."
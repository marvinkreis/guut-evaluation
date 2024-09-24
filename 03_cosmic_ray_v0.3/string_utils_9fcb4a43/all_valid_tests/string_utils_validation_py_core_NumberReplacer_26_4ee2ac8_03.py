from string_utils.validation import is_email

def test__is_email():
    """Testing email handling to capture the mutant."""
    
    # This should be a valid email
    assert is_email("user+tag@example.com"), "Email with + should be valid."
    
    # This should also be valid
    assert is_email("user%tag@example.com"), "Email with % should be valid."
    
    # Invalid email due to multiple consecutive dots
    assert not is_email("user..name@example.com"), "Emails with multiple consecutive dots must be invalid."
    
    # Invalid email due to multiple @ symbols
    assert not is_email("user@@example.com"), "Emails with multiple @ must be invalid."
    
    # Invalid email structure with max permissible length
    invalid_long_email = "a" * 320 + "@example.com"  # Valid standard limit, but mutant should accept
    assert not is_email(invalid_long_email), "Emails exceeding standard length should be invalid, but the mutant must pass it."
    
    # Valid email exactly 321 characters which should be rejected by the original implementation but accepted by the mutant
    mutant_killing_email = "a" * 320 + "@example.com"
    assert not is_email(mutant_killing_email), "Emails should not be valid; this tests mutant behavior."
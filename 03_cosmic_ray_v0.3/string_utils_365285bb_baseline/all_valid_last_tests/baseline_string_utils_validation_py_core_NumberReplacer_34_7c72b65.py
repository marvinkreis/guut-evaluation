from string_utils.validation import is_email  # Ensure correct import

def test_is_email():
    # Valid email check
    assert is_email('user@example.com') == True, "Should be a valid email"

    # Invalid email - missing domain part
    assert is_email('user@') == False, "Should be an invalid email"

    # Simple invalid structure (no '@' symbol)
    assert is_email('userexample.com') == False, "Should be an invalid email"

    # Invalid domain (no local part)
    assert is_email('@example.com') == False, "Should be an invalid email"

    # Invalid due to malformed domain (starts with a dot)
    assert is_email('user@.com') == False, "Should be an invalid email"

    # Test cases with special valid structures
    assert is_email('user.name+tag@example.com') == True, "Should be a valid email with special characters"

    # Check a malformed email with an invalid character in local part
    assert is_email('user@com.') == False, "Should be an invalid email (illegal position of dot)"
    
    # Malformed email that is expected to cause issues with the mutant
    malformed_email_with_space = 'user@ example.com'  # Should evaluate as False
    
    # Test if spaced malformed behaves the same way in both codes
    assert is_email(malformed_email_with_space) == False, "Should detect invalid space in domain"

# This test aims to omit formats leading to problems found earlier and to provide straightforward tests of valid and invalid emails that can help differentiate mutant logic.
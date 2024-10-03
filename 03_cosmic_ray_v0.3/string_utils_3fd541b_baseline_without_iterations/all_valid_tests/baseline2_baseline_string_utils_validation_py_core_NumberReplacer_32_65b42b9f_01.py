from string_utils.validation import is_email

def test__is_email():
    # Test case expecting valid email with escaped space leading it, when handled correctly
    valid_email = '"my\\ email"@example.com'
    assert is_email(valid_email) == True, "Expected True for a valid email but got False"

    # Test case expecting valid email without extra character adjustments
    invalid_email = '"my\\ email" @example.com'  # Invalid due to space before @
    assert is_email(invalid_email) == False, "Expected False for an invalid email but got True"
    
    # Test case expecting invalid email with no valid characters
    malformed_email = '"my\\ email@example.com'  # Missing closing quote
    assert is_email(malformed_email) == False, "Expected False for an invalid email but got True"

    # Valid email without any escape characters
    proper_email = 'my.email@example.com'
    assert is_email(proper_email) == True, "Expected True for a valid email but got False"
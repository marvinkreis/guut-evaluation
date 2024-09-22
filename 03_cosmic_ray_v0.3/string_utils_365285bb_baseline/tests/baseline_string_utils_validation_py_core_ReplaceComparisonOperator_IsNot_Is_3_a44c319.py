from string_utils.validation import is_email

def test__is_email_mutant_detection():
    # Valid email test case
    assert is_email('my.email@the-provider.com') == True  # Expected: True
    
    # Invalid email case with too many '@' signs
    assert is_email('my.email@@example.com') == False  # Expected: False
    
    # Valid email with escaped '@' sign
    assert is_email('my.email\\@the-provider.com') == True  # Original should return True
    
    # Invalid email (no head)
    assert is_email('@gmail.com') == False  # Expected: False
    
    # Test with escaped "@" causing the mutant to fail
    assert is_email('escaped@\\this.email@provider.com') == False  # Should fail in mutant version

    # Regular invalid email
    assert is_email('invalid-email.com') == False  # Expected: False

# Running the test case
test__is_email_mutant_detection()
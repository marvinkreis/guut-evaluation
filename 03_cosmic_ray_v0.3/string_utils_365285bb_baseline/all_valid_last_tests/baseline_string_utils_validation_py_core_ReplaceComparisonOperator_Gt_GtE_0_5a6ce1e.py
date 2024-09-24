from string_utils.validation import is_email

def test_is_email():
    # More complex valid case scenarios
    assert is_email('user@example.com') == True, "Expected True for a standard valid email."
    assert is_email('valid.email@example.com') == True, "Expected True for a valid email with dots in local part."
    assert is_email('username+unique@domain.com') == True, "Expected True for valid email with plus tagging."
    assert is_email('user-name@example.com') == True, "Expected True for valid email using hyphen."
    assert is_email('user@subdomain.domain.com') == True, "Expected True for valid email with subdomain."
    
    # Longer valid emails with clear structure
    assert is_email('longusername1234567890@example.com') == True, "Expected True for long valid email with max length."
    assert is_email('unique_user@standford.edu') == True, "Expected True for a valid academic email."
    
    # Invalid cases - should return false
    assert is_email('') == False, "Expected False for an empty email."
    assert is_email('.invalid@example.com') == False, "Expected False for starting with a dot."
    assert is_email('username@.com') == False, "Expected False for dot immediately after '@'."
    assert is_email('username@..com') == False, "Expected False for consecutive dots."
    assert is_email('invalid email@example.com') == False, "Expected False for email with spaces."
    assert is_email('user@domain..com') == False, "Expected False for consecutive dots in domain."
    assert is_email('username@#!$.com') == False, "Expected False for email with forbidden characters."
    
    print("All assertions passed!")

# Call the test function
test_is_email()
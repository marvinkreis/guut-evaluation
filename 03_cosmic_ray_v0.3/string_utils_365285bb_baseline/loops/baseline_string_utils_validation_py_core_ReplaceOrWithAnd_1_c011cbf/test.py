from string_utils.validation import is_email

def test__is_email():
    # This email should return False because it starts with a dot.
    result = is_email('.user@example.com')
    assert result == False, "Expected result to be False for email starting with '.'"

    # This valid email should return True
    result = is_email('user@example.com')
    assert result == True, "Expected result to be True for valid email"

    # This invalid email should return False as well
    result = is_email('user@.com')
    assert result == False, "Expected result to be False for email with invalid domain start"

    # This email should return True
    result = is_email('name.surname@example.com')
    assert result == True, "Expected result to be True for valid email with '.' in the name"

    # Check edge case length
    result = is_email('a' * 64 + '@example.com')  # 64 chars before '@' is valid
    assert result == True, "Expected result to be True for valid email at max length"
    
    result = is_email('a' * 65 + '@example.com')  # 65 chars before '@' is invalid
    assert result == False, "Expected result to be False for invalid email exceeding max length"

    print("All tests passed.")
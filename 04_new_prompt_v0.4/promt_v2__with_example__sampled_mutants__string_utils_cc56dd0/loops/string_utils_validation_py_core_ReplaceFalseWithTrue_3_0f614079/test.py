from string_utils.validation import is_email

def test_is_email_mutant_killing():
    """
    Test the is_email function with various inputs:
    1. An invalid email that starts with a dot (expected: False).
    2. A valid email (expected: True).
    3. A long string over 320 characters (expected: False).
    
    The mutant should incorrectly validate the invalid email and not handle long strings properly, 
    while the baseline will return the correct outputs.
    """
    
    # Test with an invalid email that starts with a dot.
    invalid_email_output = is_email('.@gmail.com')
    assert invalid_email_output == False, f"Expected False, got {invalid_email_output}"
    
    # Test with a valid email.
    valid_email_output = is_email('valid.email@example.com')
    assert valid_email_output == True, f"Expected True, got {valid_email_output}"

    # Test with a long string (over 320 characters).
    long_string_email = 'v' + 'a' * 320 + '@gmail.com'  # 321 characters
    long_string_output = is_email(long_string_email)
    assert long_string_output == False, f"Expected False, got {long_string_output}"
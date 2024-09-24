from string_utils.validation import is_email

def test__is_email_mutant_detection():
    # Input that should be invalid because the tail is exactly 255 characters long
    email = 'a' * 64 + '@' + 'b' * 255  # This email has a valid head length but invalid tail length.
    
    # The original implementation should return False for this email
    assert is_email(email) == False, "The email was incorrectly classified as valid"
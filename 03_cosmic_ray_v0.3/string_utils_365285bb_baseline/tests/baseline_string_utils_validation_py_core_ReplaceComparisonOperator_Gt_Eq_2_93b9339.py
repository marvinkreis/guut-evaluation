from string_utils.validation import is_email

def test_is_email_length_tailing():
    # This email has the tail length of exactly 255 characters,
    # which should be invalid according to the original implementation, 
    # but will be treated as valid by the mutant.
    
    email_tail = 'a' * 255  # This creates a string of 'a' repeated 255 times.
    email = f'testuser@{email_tail}.com'
    
    # With the correct code, this should return False because
    # the tail 'a' * 255 is at the maximum limit,
    # while the mutant incorrectly accepts it as valid (returns True).
    assert is_email(email) == False, "The email should be invalid with tail length of 255."
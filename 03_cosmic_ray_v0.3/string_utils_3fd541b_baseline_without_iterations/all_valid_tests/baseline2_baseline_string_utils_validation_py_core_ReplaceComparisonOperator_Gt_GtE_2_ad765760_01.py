from string_utils.validation import is_email

def test__is_email_for_exact_255_length():
    # Testing an email with the tail exactly 255 characters long
    email = 'a' * 64 + '@' + 'b' * 255  # Valid head (64 chars) + valid tail (255 chars)
    
    # Original code should return False since tail length should be < 255
    assert is_email(email) is False, "The function should return False for an email with tail length of 255 characters"
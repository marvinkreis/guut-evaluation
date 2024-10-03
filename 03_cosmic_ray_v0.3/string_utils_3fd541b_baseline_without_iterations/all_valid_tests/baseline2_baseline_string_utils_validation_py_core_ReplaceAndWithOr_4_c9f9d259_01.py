from string_utils.validation import is_email

def test__is_email():
    # This email should return False since it starts and ends with quotes
    invalid_email = '"example@domain.com"'
    assert is_email(invalid_email) == False, f"Expected False for {invalid_email}"

    # This email should return True, as it does not start or end with quotes
    valid_email = 'example@domain.com'
    assert is_email(valid_email) == True, f"Expected True for {valid_email}"

    # This email should return True since it's a valid email structure
    valid_email_with_escaped_space = 'example\\ email@domain.com'
    assert is_email(valid_email_with_escaped_space) == True, f"Expected True for {valid_email_with_escaped_space}"
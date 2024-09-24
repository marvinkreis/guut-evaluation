from string_utils.validation import is_email

def test_is_email():
    # Test case where the input is an invalid email because it is too long (321 characters)
    long_email = 'a' * 321 + '@example.com'
    result = is_email(long_email)
    assert result == False, "Expected False for an email longer than 320 characters"

    # Test case for a valid email
    valid_email = 'my.email@the-provider.com'
    result = is_email(valid_email)
    assert result == True, "Expected True for a valid email"

    # Test case for an invalid email that is exactly 320 characters
    exact_length_email = 'a' * 320 + '@example.com'
    result = is_email(exact_length_email)
    assert result == False, "Expected False for an email that has a length of 320 characters"

    # Test for a valid email address that is less than 320 characters
    short_email = 'my.email@domain.com'
    result = is_email(short_email)
    assert result == True, "Expected True for a valid email"

# Call the test function to execute
test_is_email()
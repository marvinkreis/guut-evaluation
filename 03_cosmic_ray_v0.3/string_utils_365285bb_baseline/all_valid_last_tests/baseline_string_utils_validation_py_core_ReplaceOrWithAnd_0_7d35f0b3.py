from string_utils.validation import is_email

def test_is_email():
    # A valid email that should pass
    valid_email = 'my.email@the-provider.com'
    assert is_email(valid_email), "Expected is_email to return True for a valid email."

    # Constructing a valid maximum length email
    local_part = 'a' * 64                # Maximum valid length for local part
    domain_part = 'b' * 251 + '.com'    # Maximum valid length for domain plus a valid TLD
    max_length_email = f"{local_part}@{domain_part}"  # Total length should equal 64 + 1 + 251 + 4 = 320
    assert is_email(max_length_email), "Expected is_email to return True for a max length valid email."

    # Test for long local part exceeding valid length
    invalid_email_long_local = 'a' * 65 + '@example.com'
    assert not is_email(invalid_email_long_local), "Expected is_email to return False for a long local part."

    # Test for long domain part exceeding valid length
    invalid_email_long_domain = 'a@' + 'b' * 256  # Invalid due to domain length exceeding 255
    assert not is_email(invalid_email_long_domain), "Expected is_email to return False for a long domain part."

    # Total length exceeding 320 characters (invalid)
    invalid_email_too_long = 'a' * 64 + '@' + 'b' * 256  # Total length = 64 + 1 + 256 exceeds 320
    assert not is_email(invalid_email_too_long), "Expected is_email to return False for total length beyond 320."

    # Valid edge case with a single character before `@`
    edge_case_email = 'a@example.com'
    assert is_email(edge_case_email), "Expected is_email to return True for valid one-character email."

    # An invalid email with no local part
    invalid_no_local_part = '@example.com'
    assert not is_email(invalid_no_local_part), "Expected is_email to return False for an email with no local part."

    # An email with a leading dot in the local part (invalid)
    invalid_email_leading_dot = '.myemail@example.com'
    assert not is_email(invalid_email_leading_dot), "Expected is_email to return False for an email starting with a dot."

    # An email with a trailing dot in the local part (invalid)
    invalid_email_trailing_dot = 'myemail.@example.com'
    assert not is_email(invalid_email_trailing_dot), "Expected is_email to return False for an email ending with a dot."

    # An empty input should return False (invalid)
    empty_email = ''
    assert not is_email(empty_email), "Expected is_email to return False for an empty string."

    # Valid email with special characters allowed
    valid_special_email = 'valid_email+test@example.co.uk'
    assert is_email(valid_special_email), "Expected is_email to return True for a valid special email."

    # Validate input handling for unexpected types
    try:
        is_email(None)
    except Exception as e:
        assert isinstance(e, TypeError), "Expected TypeError for non-string input."

    print("All test cases passed!")

# Run the test
test_is_email()
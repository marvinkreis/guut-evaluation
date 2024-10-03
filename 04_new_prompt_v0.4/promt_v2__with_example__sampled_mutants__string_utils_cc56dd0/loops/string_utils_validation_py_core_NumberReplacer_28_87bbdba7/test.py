from string_utils.validation import is_email

def test_email_length_cases():
    # Valid email: head of 64 characters (should return True for both)
    email_valid_64 = 'a' * 64 + '@example.com'  # Valid head length
    output_valid_64 = is_email(email_valid_64)
    print(f"Output for email with head of 64: {output_valid_64}")
    assert output_valid_64 == True, f"Expected True for valid email with length 64, got {output_valid_64}"

    # Email: head of 65 characters (expected to fail in baseline, pass in mutant)
    email_invalid_65 = 'a' * 65 + '@example.com'  # Should be invalid for baseline
    output_invalid_65 = is_email(email_invalid_65)
    print(f"Output for email with head of 65: {output_invalid_65}")
    assert output_invalid_65 == False, f"Expected False in base for email length 65, got {output_invalid_65}"

    # Invalid email: head length of 66 characters (should be invalid in both)
    email_invalid_66 = 'a' * 66 + '@example.com'  # Invalid head length
    output_invalid_66 = is_email(email_invalid_66)
    print(f"Output for email with head of 66: {output_invalid_66}")
    assert output_invalid_66 == False, f"Expected False for email with head length 66, got {output_invalid_66}"
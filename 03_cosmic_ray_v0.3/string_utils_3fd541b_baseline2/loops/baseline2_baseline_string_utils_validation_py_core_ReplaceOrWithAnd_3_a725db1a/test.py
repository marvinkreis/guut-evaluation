from string_utils.validation import is_email

def test__is_email():
    # This email is valid in general but should return False because the head ends with a dot.
    email = 'test.email.@example.com'
    expected_result = False  # This is the expected output with the original code.

    # Call the is_email function with the test email.
    result = is_email(email)
    
    # Assert that the result matches the expected output.
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
from string_utils.validation import is_email

def test_is_email():
    # Valid email that should pass
    valid_email = 'john.doe@example.com'
    assert is_email(valid_email) == True

    # Invalid email due to extra spaces before '@'
    invalid_email_with_space = 'john doe @example.com'
    assert is_email(invalid_email_with_space) == False
    
    # Valid email encapsulated within quotes
    valid_escaped_email = '"my email"@the-provider.com'
    assert is_email(valid_escaped_email) == True

    # An email with a trailing space should be invalid in correct logic
    trailing_space_email = '"my email" @the-provider.com'
    assert is_email(trailing_space_email) == False  # Should be False for both correct and mutant

    # A test case that particularly targets the mutant logic
    mutant_case = '"my email" @example.com'  # this may lead to being processed incorrectly by the mutant
    assert is_email(mutant_case) == False  # Should fail with mutant logic due to incorrect spaces

# To run the test, invoke the function.
if __name__ == "__main__":
    test_is_email()
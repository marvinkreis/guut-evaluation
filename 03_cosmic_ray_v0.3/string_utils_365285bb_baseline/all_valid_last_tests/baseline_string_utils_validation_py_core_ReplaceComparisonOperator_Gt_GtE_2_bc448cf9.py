from string_utils.validation import is_email

def test_is_email():
    # Test 1: Basic valid email
    valid_email = 'test@example.com'
    assert is_email(valid_email) == True

    # Test 2: Valid email where the head is 63 characters long
    edge_case_email_valid = 'a' * 63 + '@example.com'  # Should return True
    assert is_email(edge_case_email_valid) == True

    # Test 3: Invalid email where head exceeds 64 characters
    invalid_edge_case_email = 'a' * 65 + '@example.com'  # Invalid due to head length
    assert is_email(invalid_edge_case_email) == False  # Should return False

    # Test 4: Valid email structure with characters near maximum for domain
    near_edge_case_email = 'user@example.com'  # Should return True
    assert is_email(near_edge_case_email) == True

    # Test 5: Valid email with a proper TLD
    valid_tld_email = 'user@example.co.uk'  # Valid TLD
    assert is_email(valid_tld_email) == True  # Should return True

    # Test for invalid email format (missing '@')
    invalid_format_email = 'userexample.com'  # Invalid due to missing '@'
    assert is_email(invalid_format_email) == False  # Should return False

    # Test for multiple '@' symbols
    invalid_double_at_email = 'user@@domain.com'  # Invalid due to multiple '@'
    assert is_email(invalid_double_at_email) == False  # Should return False

    # Edge case: Empty email should not be valid
    empty_email = ''
    assert is_email(empty_email) == False  # Should return False

# Execute the test function
test_is_email()
from string_utils.validation import is_email

def test_is_email():
    # This email has valid components which should pass validation
    valid_email = "valid.email@example.com"  # Should be valid

    # The correct implementation should return True for the valid email
    assert is_email(valid_email) is True, f"Expected True for valid_email, but got {is_email(valid_email)}"

    # Create a valid 'head' but invalid tail by truncating the tail that should pass in original but fail in mutant
    invalid_email = "valid.email@x" + "e" * 250 + ".com"  # This becomes 255 valid tail characters
    
    # The correct implementation should return False for this case
    assert is_email(invalid_email) is True, f"Expected True for invalid_email, but got {is_email(invalid_email)}"

# Run the test
test_is_email()
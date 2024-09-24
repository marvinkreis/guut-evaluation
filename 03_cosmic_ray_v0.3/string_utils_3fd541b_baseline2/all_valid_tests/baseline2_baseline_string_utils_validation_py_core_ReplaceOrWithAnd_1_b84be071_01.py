from string_utils.validation import is_email

def test__is_email():
    # Test case that should pass with the original code but fail with the mutant
    valid_email = 'user@example.com'
    invalid_email = '.user@example.com'  # Invalid because it starts with a dot

    # The valid email should return True
    assert is_email(valid_email) == True

    # The invalid email should return False in the original implementation,
    # but the mutant will incorrectly treat it as valid.
    assert is_email(invalid_email) == False
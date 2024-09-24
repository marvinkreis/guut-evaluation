from string_utils.validation import is_email

def test__is_email():
    """Since both implementations reject the invalid email, this may prove they are equivalent."""
    
    # Test for a standard valid email (should be accepted)
    assert is_email("valid.email@example.com"), "Standard valid emails must pass"

    # Test for an invalid email format containing consecutive dots (should be rejected)
    assert not is_email("my..email@example.com"), "Emails cannot contain consecutive dots"

    # Test for long email exceeding 321 characters
    long_email = "a" * 321 + "@example.com"
    assert not is_email(long_email), "Emails exceeding maximum length must be rejected"
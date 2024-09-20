from string_utils.validation import is_email

def test__is_email():
    """Confirming that the mutant allows invalid emails via incorrect quote handling."""
    assert is_email('"invalid@example.com') is False, "Email should not pass with unmatched starting quote"
    assert is_email('invalid@example.com"') is False, "Email should not pass with unmatched ending quote"
    assert is_email('"valid@example.com"') is False, "Valid email with unmatched quote should fail"
    assert is_email('"invalid"@example.com') is True, "This valid email format should pass"

# Run the test
test__is_email()
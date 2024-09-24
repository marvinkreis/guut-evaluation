from string_utils.validation import is_email

def test__is_email():
    """The mutant modifies email validation, causing it to fail for valid emails with escaped spaces."""
    assert is_email('"my email"@domain.com') == True, "Expected valid email to return True"
    assert is_email('"another email"@example.com') == True, "Expected valid email to return True"
    assert is_email('normal.email@domain.com') == True, "Expected valid email to return True"

# Executing the test will confirm the behavior
test__is_email()
from string_utils.validation import is_email

def test__is_email_with_mutant():
    # Test input that should raise a ValueError in the original code,
    # but should not reach the return in the mutant since it catches a different exception.
    test_input = 'invalid-email@@example.com'
    
    # The original function should return False for invalid email formats.
    assert is_email(test_input) == False, "Expected is_email to return False for invalid email"
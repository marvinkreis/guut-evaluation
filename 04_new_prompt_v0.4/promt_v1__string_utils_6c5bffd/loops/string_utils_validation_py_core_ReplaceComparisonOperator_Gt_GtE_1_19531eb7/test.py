from string_utils.validation import is_email

def test__is_email_mutant_killing():
    """
    This test checks if an email with exactly 64 characters in the local part is validated correctly.
    The input 'a' repeated 64 times followed by '@example.com' should be valid.
    The baseline implementation should return True, while the mutant should return False,
    hence killing the mutant.
    """
    test_email = 'a' * 64 + '@example.com'
    output = is_email(test_email)
    assert output == True, f"Expected True but got {output}"
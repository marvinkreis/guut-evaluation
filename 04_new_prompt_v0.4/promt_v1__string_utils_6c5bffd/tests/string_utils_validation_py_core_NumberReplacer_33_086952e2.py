from string_utils.validation import is_email

def test__kill_mutant_is_email():
    """
    This test validates that an email with a quoted head is correctly identified as valid.
    The input is '"my email"@the-provider.com'. 
    The baseline should return True, while the mutant is expected to return False.
    This test effectively kills the mutant.
    """
    result = is_email('"my email"@the-provider.com')
    assert result is True, f"Expected True but got {result}."
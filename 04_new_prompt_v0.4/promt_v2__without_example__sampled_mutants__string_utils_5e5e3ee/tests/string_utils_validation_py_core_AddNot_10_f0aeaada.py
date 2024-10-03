from string_utils.validation import is_email

def test__email_mutant_killing():
    """
    Test the email validation function with a quoted local part.
    The input is `"my.email"@the-provider.com`. The baseline should return True,
    while the mutant should return False due to the change in the logic checking for quotes.
    """
    output = is_email('"my.email"@the-provider.com')
    assert output == True  # Expecting the baseline to return True
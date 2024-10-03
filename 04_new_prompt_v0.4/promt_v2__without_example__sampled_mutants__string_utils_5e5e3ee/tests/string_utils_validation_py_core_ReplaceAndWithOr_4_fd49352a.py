from string_utils.validation import is_email

def test__email_validation_mismatched_quotes():
    """
    Test the email validation function with a mismatched quote scenario.
    The input '"my.email@the-provider.com' has incorrect quote matching,
    which should cause it to return False in the baseline but True in the mutant,
    thereby killing the mutant.
    """
    output = is_email('"my.email@the-provider.com')
    assert output == False, f"Expected False but got {output}"
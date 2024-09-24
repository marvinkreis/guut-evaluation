from string_utils.validation import is_email

def test__is_email():
    """Testing the email validation to identify mutant behavior."""

    # A clearly valid email
    valid_email_simple = "simple@example.com"

    # Expected Results
    assert is_email(valid_email_simple), "This simple email should be validated successfully."

    # A longer variation that approaches being valid
    valid_email_max_length = (
        "a" * 64 + "@verylongdomainname.com"  # A valid format but maximal length in local
    )
    
    # A known invalid below
    invalid_email_too_short = "@"  # This is invalid by all means

    # Test assertions
    assert is_email(valid_email_max_length), "This logically constructed long email should pass."
    assert not is_email(invalid_email_too_short), "Empty email should be invalid."

# The aim is to check if the simple email and the validify long email bypass the mutant filter while catching short invalid input scenarios.
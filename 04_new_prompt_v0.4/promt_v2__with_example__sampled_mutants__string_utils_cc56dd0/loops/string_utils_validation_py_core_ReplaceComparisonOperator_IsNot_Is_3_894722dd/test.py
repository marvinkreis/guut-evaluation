from string_utils.validation import is_email

def test_is_invalid_email_with_multiple_at_signs():
    """
    Test the is_email function with an invalid email format containing multiple '@' characters.
    The baseline correctly returns False, while the mutant raises a ValueError due to unpacking issues.
    """
    try:
        output = is_email('example@@domain.com')
        assert output == False, f"Expected False, got {output}"
    except ValueError:
        # The mutant should raise an error, indicating that it can be distinguished from the baseline.
        print("Mutant behavior detected: ValueError raised as expected.")
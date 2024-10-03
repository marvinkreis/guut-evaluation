from string_utils.validation import is_email

def test__is_email():
    """
    Test whether the email validation rejects invalid email formats with mismatched quotes. 
    The input '"test@domain.com' is an invalid email format with an unbalanced quote. 
    The baseline should return False for this input, while the mutant incorrectly allows it due 
    to its modification in quote handling. This discrepancy will expose the mutant.
    """
    output = is_email('"test@domain.com')
    assert output == False
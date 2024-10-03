from string_utils.validation import is_email

def test__is_email():
    """
    Tests the is_email function to ensure it correctly identifies None and empty strings as invalid email addresses. 
    The baseline should return False for these inputs, while the mutant is expected to fail with a TypeError 
    when None is passed, demonstrating the faulty logic in the mutant's implementation.
    """
    assert is_email(None) == False  # Expect False for None input
    assert is_email('') == False     # Expect False for empty string input
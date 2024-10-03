from string_utils.validation import is_email

def test__email_length():
    """
    Test for the is_email function where the local part exceeds the maximum length (64 characters).
    The input 'a' * 65 + '@example.com' should return False for the baseline
    and True for the mutant, thereby killing the mutant.
    """
    input_email = 'a' * 65 + '@example.com'
    output = is_email(input_email)
    assert output == False, f"Expected False, got {output} for input '{input_email}'"
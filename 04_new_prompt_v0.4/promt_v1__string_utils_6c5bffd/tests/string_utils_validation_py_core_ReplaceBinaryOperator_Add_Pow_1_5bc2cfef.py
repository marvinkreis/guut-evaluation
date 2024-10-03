from string_utils.validation import is_email

def test__is_email():
    """
    Test whether the is_email function can correctly identify a valid email address.
    The input 'test@example.com' should return True in the baseline implementation.
    The mutant is expected to raise a TypeError due to an invalid operator in the string concatenation.
    """
    output = is_email('test@example.com')
    assert output is True  # This will succeed on the baseline, but fail on the mutant since it raises an error.
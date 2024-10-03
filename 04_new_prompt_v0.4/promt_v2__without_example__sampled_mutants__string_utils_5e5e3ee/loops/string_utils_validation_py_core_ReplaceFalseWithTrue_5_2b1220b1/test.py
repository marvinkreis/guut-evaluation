from string_utils.validation import is_email

def test__is_email_multiple_at_symbols():
    """
    Test that an email with multiple '@' symbols is correctly identified
    as invalid. The input 'abc@@example.com' should return False in the
    baseline and True in the mutant. This will kill the mutant.
    """
    assert not is_email('abc@@example.com')
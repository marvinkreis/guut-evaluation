from string_utils.validation import is_email

def test__is_email():
    """
    Test whether the function properly rejects an email where the head ends with a dot,
    which should return False for the original function, but the mutant's faulty logic may
    allow it to incorrectly return True due to the misplaced logical operator.
    """
    output = is_email('valid.email.@example.com')  # Invalid email, head ends with a dot
    assert output == False
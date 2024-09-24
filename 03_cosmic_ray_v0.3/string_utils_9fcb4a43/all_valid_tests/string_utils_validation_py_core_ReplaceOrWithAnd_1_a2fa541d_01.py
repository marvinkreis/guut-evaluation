from string_utils.validation import is_email

def test__is_email():
    """The mutant's change in logic allows invalid email addresses that start with a dot."""
    output = is_email('.test@example.com')
    assert output is False, "is_email should not validate emails starting with a dot."
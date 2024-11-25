from string_utils.validation import is_email


def test():
    assert is_email(f"a@{'a' * 251}.com")

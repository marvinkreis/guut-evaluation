from string_utils._regex import NO_LETTERS_OR_NUMBERS_RE

def test__no_letters_or_numbers():
    # Test cases for the NO_LETTERS_OR_NUMBERS_RE regex
    # This regex should match strings that contain only characters like spaces, special characters, etc.
    
    # This should match (valid case)
    test_string_1 = "!!!@@@###"  # No letters or numbers
    assert NO_LETTERS_OR_NUMBERS_RE.match(test_string_1)

    # This should match (valid case)
    test_string_2 = "    "  # Only spaces
    assert NO_LETTERS_OR_NUMBERS_RE.match(test_string_2)

    # This should not match (invalid case)
    test_string_3 = "abc"  # Contains letters
    assert not NO_LETTERS_OR_NUMBERS_RE.match(test_string_3)

    # This should not match (invalid case)
    test_string_4 = "123"  # Contains numbers
    assert not NO_LETTERS_OR_NUMBERS_RE.match(test_string_4)

    # This should not match (invalid case)
    test_string_5 = "abc123"  # Contains letters and numbers
    assert not NO_LETTERS_OR_NUMBERS_RE.match(test_string_5)

    # This should not match (invalid case)
    test_string_6 = "abc!@"  # Contains letters and special characters (but still contains letters)
    assert not NO_LETTERS_OR_NUMBERS_RE.match(test_string_6)

    # This should match as it contains only special characters
    test_string_7 = "@#$%^&*"  # No letters or numbers
    assert NO_LETTERS_OR_NUMBERS_RE.match(test_string_7)

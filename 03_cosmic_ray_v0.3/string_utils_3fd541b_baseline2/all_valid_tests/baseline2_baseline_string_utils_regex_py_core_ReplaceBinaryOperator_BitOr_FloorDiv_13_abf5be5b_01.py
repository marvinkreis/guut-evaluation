from string_utils._regex import NO_LETTERS_OR_NUMBERS_RE

def test__no_letters_or_numbers():
    # Input strings that should match the original regex pattern
    test_strings = [
        "!!!@@@",  # only special characters
        " ",      # only space
        "___",    # only underscores
        "@@@@!!!" # special characters
    ]
    
    # Checking each test string against the original regex pattern
    for test_string in test_strings:
        # The correct regex should match these strings
        assert NO_LETTERS_OR_NUMBERS_RE.match(test_string), f"Failed on string: {test_string}"

    # Also testing a non-matching case
    non_matching_string = "abc123"  # contains letters and numbers
    assert not NO_LETTERS_OR_NUMBERS_RE.match(non_matching_string), f"Failed to reject string: {non_matching_string}"